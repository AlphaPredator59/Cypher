"""
CYPHER Intrusion Detection System — Database & Persistence Module.
Provides SQLAlchemy 2.0 ORM models, connection lifecycle management,
and indexed telemetry persistence for network flow predictions and alerts.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, create_engine, select, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

# Database configuration: defaults to artifacts/cypher.db or CYPHER_DB_PATH / CYPHER_DATABASE_URL
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_FILE = os.path.join(PROJECT_ROOT, "artifacts", "cypher.db")
DB_PATH = os.getenv("CYPHER_DB_PATH", DEFAULT_DB_FILE)
DATABASE_URL = os.getenv("CYPHER_DATABASE_URL", f"sqlite:///{DB_PATH}")

# Ensure parent directory exists for SQLite
if DATABASE_URL.startswith("sqlite:///"):
    sqlite_file = DATABASE_URL.replace("sqlite:///", "")
    if sqlite_file and sqlite_file != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(sqlite_file)), exist_ok=True)

# Engine configuration
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)


class Base(DeclarativeBase):
    pass


class FlowAuditRecord(Base):
    """
    Persisted network flow inspection record.
    Stores factual telemetry metadata and model outputs without raw packet payloads.
    Indexed on timestamp, src_ip, dst_ip, is_anomaly, and detection_source.
    """
    __tablename__ = "flow_audit_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    flow_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )
    src_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True, index=True)
    dst_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True, index=True)
    src_port: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    dst_port: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    protocol: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)

    # Deterministic VAE anomaly signals
    score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    anomaly_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Supervised classification signals
    attack_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    attack_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    classification_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    # Verdict fusion signals
    verdict: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    verdict_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    detection_source: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)

    __table_args__ = (
        Index("ix_flow_src_dst_ip", "src_ip", "dst_ip"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "flow_id": self.flow_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "score": self.score,
            "threshold": self.threshold,
            "is_anomaly": self.is_anomaly,
            "anomaly_confidence": self.anomaly_confidence,
            "attack_type": self.attack_type,
            "attack_confidence": self.attack_confidence,
            "classification_status": self.classification_status,
            "verdict": self.verdict,
            "verdict_confidence": self.verdict_confidence,
            "detection_source": self.detection_source,
        }


def init_db(target_engine=None) -> None:
    """Creates database tables if they do not exist."""
    e = target_engine or engine
    Base.metadata.create_all(bind=e)


@contextmanager
def get_db_session(session_factory=None) -> Generator[Session, None, None]:
    """Context manager for scoped database sessions with automatic commit/rollback."""
    factory = session_factory or SessionLocal
    db: Session = factory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def save_flow_record(flow_data: Dict[str, Any], db: Optional[Session] = None) -> FlowAuditRecord:
    """
    Persists a single flow inspection record to the database.
    """
    record = FlowAuditRecord(
        flow_id=str(flow_data.get("flow_id") or flow_data.get("row_index") or ""),
        src_ip=flow_data.get("src_ip"),
        dst_ip=flow_data.get("dst_ip"),
        src_port=int(flow_data["src_port"]) if flow_data.get("src_port") is not None else None,
        dst_port=int(flow_data["dst_port"]) if flow_data.get("dst_port") is not None else None,
        protocol=str(flow_data.get("protocol") or "TCP/IP"),
        score=float(flow_data.get("score", 0.0)),
        threshold=float(flow_data.get("threshold", 38.50959)),
        is_anomaly=bool(flow_data.get("is_anomaly", False)),
        anomaly_confidence=float(flow_data["anomaly_confidence"]) if flow_data.get("anomaly_confidence") is not None else None,
        attack_type=flow_data.get("attack_type"),
        attack_confidence=float(flow_data["attack_confidence"]) if flow_data.get("attack_confidence") is not None else None,
        classification_status=flow_data.get("classification_status"),
        verdict=flow_data.get("verdict"),
        verdict_confidence=float(flow_data["verdict_confidence"]) if flow_data.get("verdict_confidence") is not None else None,
        detection_source=flow_data.get("detection_source"),
    )

    if db is not None:
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    with get_db_session() as session:
        session.add(record)
        session.commit()
        session.refresh(record)
        session.expunge(record)
        return record


def save_flow_records(flows: List[Dict[str, Any]], db: Optional[Session] = None) -> int:
    """
    Bulk-persists a list of flow records to the database.
    Returns the number of saved records.
    """
    if not flows:
        return 0

    records = [
        FlowAuditRecord(
            flow_id=str(f.get("flow_id") or f.get("row_index") or ""),
            src_ip=f.get("src_ip"),
            dst_ip=f.get("dst_ip"),
            src_port=int(f["src_port"]) if f.get("src_port") is not None else None,
            dst_port=int(f["dst_port"]) if f.get("dst_port") is not None else None,
            protocol=str(f.get("protocol") or "TCP/IP"),
            score=float(f.get("score", 0.0)),
            threshold=float(f.get("threshold", 38.50959)),
            is_anomaly=bool(f.get("is_anomaly", False)),
            anomaly_confidence=float(f["anomaly_confidence"]) if f.get("anomaly_confidence") is not None else None,
            attack_type=f.get("attack_type"),
            attack_confidence=float(f["attack_confidence"]) if f.get("attack_confidence") is not None else None,
            classification_status=f.get("classification_status"),
            verdict=f.get("verdict"),
            verdict_confidence=float(f["verdict_confidence"]) if f.get("verdict_confidence") is not None else None,
            detection_source=f.get("detection_source"),
        )
        for f in flows
    ]

    if db is not None:
        db.add_all(records)
        db.commit()
        return len(records)

    with get_db_session() as session:
        session.add_all(records)
        session.commit()
        return len(records)


def query_flows_by_ip(clean_ip: str, limit: int = 100, db: Optional[Session] = None) -> List[Dict[str, Any]]:
    """
    Queries persisted flows associated with an IP address (as source or destination).
    Ordered by recency (timestamp DESC).
    """
    stmt = (
        select(FlowAuditRecord)
        .where((FlowAuditRecord.src_ip == clean_ip) | (FlowAuditRecord.dst_ip == clean_ip))
        .order_by(FlowAuditRecord.timestamp.desc())
        .limit(limit)
    )

    if db is not None:
        results = db.scalars(stmt).all()
        return [r.to_dict() for r in results]

    with get_db_session() as session:
        results = session.scalars(stmt).all()
        return [r.to_dict() for r in results]


def query_recent_flows(
    limit: int = 100,
    anomaly_only: bool = False,
    detection_source: Optional[str] = None,
    db: Optional[Session] = None
) -> List[Dict[str, Any]]:
    """
    Queries recent persisted flows with optional anomaly or detection_source filters.
    """
    stmt = select(FlowAuditRecord)
    if anomaly_only:
        stmt = stmt.where(FlowAuditRecord.is_anomaly.is_(True))
    if detection_source:
        stmt = stmt.where(FlowAuditRecord.detection_source == detection_source)

    stmt = stmt.order_by(FlowAuditRecord.timestamp.desc()).limit(limit)

    if db is not None:
        results = db.scalars(stmt).all()
        return [r.to_dict() for r in results]

    with get_db_session() as session:
        results = session.scalars(stmt).all()
        return [r.to_dict() for r in results]


def get_flow_stats(db: Optional[Session] = None) -> Dict[str, int]:
    """
    Returns aggregate counts of total, anomalous, and benign persisted flows.
    """
    if db is not None:
        total = db.scalar(select(func.count(FlowAuditRecord.id))) or 0
        anomalies = db.scalar(select(func.count(FlowAuditRecord.id)).where(FlowAuditRecord.is_anomaly.is_(True))) or 0
        return {"total": total, "anomalous": anomalies, "benign": total - anomalies}

    with get_db_session() as session:
        total = session.scalar(select(func.count(FlowAuditRecord.id))) or 0
        anomalies = session.scalar(select(func.count(FlowAuditRecord.id)).where(FlowAuditRecord.is_anomaly.is_(True))) or 0
        return {"total": total, "anomalous": anomalies, "benign": total - anomalies}
