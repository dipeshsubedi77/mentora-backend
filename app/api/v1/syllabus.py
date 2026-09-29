"""
Syllabus API endpoints
"""
import os
from typing import List, Optional

import logging
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.auth import get_current_user_id
from app.core.config import settings
from app.core.quotas import QuotaContext, record_usage, require_ai_quota
from app.database.database import get_db
from app.models.syllabus import Syllabus, Subject, Chapter
from app.models.subscription import UsageType
from app.schemas.syllabus import (
    SyllabusOut, SyllabusUpdate, SyllabusStatus,
    SyllabusSearchResponse
)
from app.services.syllabus_service import SyllabusService

router = APIRouter()
syllabus_service = SyllabusService()

logger = logging.getLogger(__name__)


async def _process_syllabus_in_background(syllabus_id: int, user_id: int) -> None:
    """OCR + LLM parsing + RAG embedding, run outside the HTTP request.

    The upload/analyze handlers schedule this via FastAPI BackgroundTasks
    so the client gets an immediate 201/200 instead of holding the
    connection open while heavy processing runs (which previously caused
    502s and worker OOMs on small instances).  The syllabus row's
    ``status`` tracks progress and callers can poll
    ``GET /api/v1/syllabus/{id}``.

    Usage is recorded only after processing succeeds, matching the old
    synchronous semantics.
    """
    from app.database.database import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as db:
            try:
                result = await db.execute(
                    select(Syllabus).where(Syllabus.id == syllabus_id)
                )
                syllabus = result.scalars().first()
                if syllabus is None:
                    logger.error(
                        "[BackgroundSyllabus] syllabus %s not found", syllabus_id
                    )
                    return

                if syllabus.status == SyllabusStatus.PROCESSING.value:
                    logger.warning(
                        "[BackgroundSyllabus] syllabus %s already processing",
                        syllabus_id,
                    )
                    return

                syllabus.status = SyllabusStatus.PROCESSING.value
                await db.commit()

                await syllabus_service.process_syllabus(db, syllabus)
                await db.commit()

                # Usage is recorded only after processing succeeded.
                await record_usage(db, user_id, UsageType.SYLLABUS_ANALYSIS)
                await db.commit()

                logger.info(
                    "[BackgroundSyllabus] syllabus %s processed (status=%s)",
                    syllabus_id,
                    syllabus.status,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[BackgroundSyllabus] processing failed for syllabus %s: %s",
                    syllabus_id,
                    exc,
                )
                await db.rollback()
                try:
                    result = await db.execute(
                        select(Syllabus).where(Syllabus.id == syllabus_id)
                    )
                    syllabus = result.scalars().first()
                    if syllabus is not None:
                        syllabus.status = SyllabusStatus.FAILED.value
                        await db.commit()
                        logger.info(
                            "[BackgroundSyllabus] marked syllabus %s as failed",
                            syllabus_id,
                        )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "[BackgroundSyllabus] failed to mark syllabus %s as failed",
                        syllabus_id,
                    )
    except Exception:  # noqa: BLE001
        logger.exception(
            "[BackgroundSyllabus] background task crashed for syllabus %s",
            syllabus_id,
        )


@router.post("/", response_model=SyllabusOut, status_code=status.HTTP_201_CREATED)
@router.post("/upload", response_model=SyllabusOut, status_code=status.HTTP_201_CREATED)
async def upload_syllabus(
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
    quota: QuotaContext = Depends(require_ai_quota(UsageType.SYLLABUS_ANALYSIS)),
):
    logger.info(
        "[UploadSyllabus] Received upload: title=%s, filename=%s, content_type=%s",
        title,
        file.filename,
        file.content_type,
    )

    file_ext = file.filename.rsplit(".", 1)[-1].lower()
    logger.info("[UploadSyllabus] Detected file_ext: %s", file_ext)

    if file_ext not in settings.ALLOWED_EXTENSIONS.split(","):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type .{file_ext} not allowed",
        )

    # Strip a trailing extension from the title so we don't end up with
    # double extensions like "syllabus.pdf.pdf".  Users often provide the
    # file name (including extension) as the title.
    safe_title = title.strip()
    if safe_title.lower().endswith(f".{file_ext}"):
        safe_title = safe_title[: -(len(file_ext) + 1)]
    safe_title = safe_title.replace(" ", "_")
    logger.info("[UploadSyllabus] safe_title: %s", safe_title)

    upload_dir = os.path.join(settings.UPLOAD_DIR, "syllabus")
    os.makedirs(upload_dir, exist_ok=True)

    # Enforce max upload size if the SpooledTemporaryFile reports a size.
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File is too large.",
        )

    file_name = f"{user_id}_{safe_title}.{file_ext}"
    file_path = os.path.join(upload_dir, file_name)

    with open(file_path, "wb") as f:
        f.write(content)

    new_syllabus = Syllabus(
        user_id=user_id,
        title=title,
        description=description,
        file_path=file_path,
        file_type=file_ext,
        status=SyllabusStatus.PROCESSING.value,
    )
    db.add(new_syllabus)
    await db.commit()
    # NOTE: intentionally skip db.refresh(new_syllabus).
    # With expire_on_commit=False the PK is already populated.

    # Respond immediately; heavy OCR/LLM/embedding work runs afterwards in
    # the background so the upload no longer 502s / OOMs small instances.
    background_tasks.add_task(
        _process_syllabus_in_background, new_syllabus.id, user_id
    )

    logger.info(
        "[UploadSyllabus] Accepted %s (id=%s) for background processing; "
        "status=%s",
        file_name,
        new_syllabus.id,
        new_syllabus.status,
    )

    # Reload the syllabus so the response model serializes cleanly.
    result = await db.execute(
        select(Syllabus)
        .where(Syllabus.id == new_syllabus.id)
        .options(
            selectinload(Syllabus.subjects).selectinload(Subject.chapters)
        )
        .execution_options(populate_existing=True)
    )
    syllabus = result.scalars().first()
    return syllabus


@router.get("/", response_model=List[SyllabusOut])
async def list_syllabuses(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    result = await db.execute(
        select(Syllabus)
        .options(selectinload(Syllabus.subjects).selectinload(Subject.chapters))
        .order_by(Syllabus.id.desc())
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()


@router.get("/search", response_model=SyllabusSearchResponse)
async def search_syllabuses(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    search_in: Optional[List[str]] = Query(
        None,
        description="Fields to search in: title, description, extracted_text, subjects, chapters, topics"
    ),
    status: Optional[SyllabusStatus] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """
    Search user's syllabuses by query string.
    
    Searches in title, description, extracted text (OCR), subject names, chapter names, and topics.
    """
    result = await syllabus_service.search_syllabuses(
        db=db,
        user_id=user_id,
        query=q,
        search_in=search_in,
        status=status,
        page=page,
        per_page=per_page,
    )
    return result


@router.get("/{syllabus_id}", response_model=SyllabusOut)
async def get_syllabus(
    syllabus_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    result = await db.execute(
        select(Syllabus)
        .options(selectinload(Syllabus.subjects).selectinload(Subject.chapters))
        .where(Syllabus.id == syllabus_id, Syllabus.user_id == user_id)
    )
    syllabus = result.scalars().first()
    if not syllabus:
        result = await db.execute(
            select(Syllabus)
            .options(selectinload(Syllabus.subjects).selectinload(Subject.chapters))
            .where(Syllabus.id == syllabus_id)
        )
        syllabus = result.scalars().first()
    if not syllabus:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Syllabus not found",
        )
    return syllabus


@router.put("/{syllabus_id}", response_model=SyllabusOut)
async def update_syllabus(
    syllabus_id: int,
    syllabus_data: SyllabusUpdate,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    result = await db.execute(
        select(Syllabus)
        .options(selectinload(Syllabus.subjects).selectinload(Subject.chapters))
        .where(Syllabus.id == syllabus_id, Syllabus.user_id == user_id)
    )
    syllabus = result.scalars().first()
    if not syllabus:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Syllabus not found",
        )

    update_data = syllabus_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(syllabus, field, value)

    db.add(syllabus)
    await db.commit()
    
    # Reload with eager loaded relationships
    result = await db.execute(
        select(Syllabus)
        .options(selectinload(Syllabus.subjects).selectinload(Subject.chapters))
        .where(Syllabus.id == syllabus.id)
    )
    return result.scalars().first()


@router.delete("/{syllabus_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_syllabus(
    syllabus_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    result = await db.execute(
        select(Syllabus).where(Syllabus.id == syllabus_id, Syllabus.user_id == user_id)
    )
    syllabus = result.scalars().first()
    if not syllabus:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Syllabus not found",
        )

    if syllabus.file_path and os.path.exists(syllabus.file_path):
        os.remove(syllabus.file_path)

    await db.delete(syllabus)
    await db.commit()
    return None


@router.post("/{syllabus_id}/analyze", response_model=SyllabusOut)
async def analyze_syllabus(
    syllabus_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
    quota: QuotaContext = Depends(require_ai_quota(UsageType.SYLLABUS_ANALYSIS)),
):
    result = await db.execute(
        select(Syllabus).where(Syllabus.id == syllabus_id, Syllabus.user_id == user_id)
    )
    syllabus = result.scalars().first()
    if not syllabus:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Syllabus not found",
        )

    # Re-processing runs in the background so the request returns quickly
    # instead of blocking on OCR/LLM/embedding (which can 502/OOM small
    # instances).  Status is poll-able via GET /api/v1/syllabus/{id}.
    syllabus.status = SyllabusStatus.PROCESSING.value
    await db.commit()

    background_tasks.add_task(_process_syllabus_in_background, syllabus_id, user_id)

    result = await db.execute(
        select(Syllabus)
        .options(selectinload(Syllabus.subjects).selectinload(Subject.chapters))
        .where(Syllabus.id == syllabus_id)
    )
    updated = result.scalars().first()
    return updated


@router.get("/{syllabus_id}/subjects")
async def get_syllabus_subjects(
    syllabus_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    result = await db.execute(
        select(Subject).join(Syllabus).where(Syllabus.id == syllabus_id)
    )
    return result.scalars().all()
