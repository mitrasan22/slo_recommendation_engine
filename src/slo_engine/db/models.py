"""
SQLAlchemy ORM models for the SLO Recommendation Engine.

Notes
-----
Uses async-first design with TimescaleDB-compatible timestamp columns.
All primary keys use UUID v4. Timestamps use ``func.now()`` server defaults
with timezone awareness. JSON columns use Python dict defaults.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """
    Declarative base class for all ORM models.

    Notes
    -----
    Provides the SQLAlchemy metaclass machinery for mapper configuration.
    All table models inherit from this class.
    """
    pass


class ServiceTier(str, PyEnum):
    """
    Enumeration of service criticality tiers.

    Attributes
    ----------
    CRITICAL : str
        User-facing, revenue-impacting services.
    HIGH : str
        Internal latency-sensitive services.
    MEDIUM : str
        Background or batch processing services.
    LOW : str
        Best-effort services with relaxed SLO requirements.

    Notes
    -----
    Used in the ``Service`` model ``tier`` column. Tier affects HITL review
    thresholds and SLO recommendation confidence requirements.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DependencyType(str, PyEnum):
    """
    Enumeration of service dependency relationship types.

    Attributes
    ----------
    SYNCHRONOUS : str
        Blocking call where failure propagates immediately.
    ASYNCHRONOUS : str
        Queued call that degrades gracefully on dependency failure.
    EXTERNAL : str
        Third-party dependency treated as unreliable without SLA.
    DATASTORE : str
        Database or cache dependency with high availability expectations.

    Notes
    -----
    Used in the ``ServiceDependency`` model ``dep_type`` column and in the
    reliability formula selection (series vs parallel).
    """

    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    EXTERNAL = "external"
    DATASTORE = "datastore"


class SLOStatus(str, PyEnum):
    """
    Enumeration of SLO lifecycle states.

    Attributes
    ----------
    DRAFT : str
        SLO is under review and not yet enforced.
    ACTIVE : str
        SLO is enforced and monitored.
    DEPRECATED : str
        SLO has been superseded by a newer version.

    Notes
    -----
    Used in the ``SLO`` model ``status`` column.
    """

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class RecommendationStatus(str, PyEnum):
    """
    Enumeration of SLO recommendation review states.

    Attributes
    ----------
    PENDING_REVIEW : str
        Recommendation is awaiting HITL review.
    APPROVED : str
        Recommendation has been approved by a reviewer.
    REJECTED : str
        Recommendation has been rejected by a reviewer.
    APPLIED : str
        Recommendation has been applied to the active SLO.

    Notes
    -----
    Used in the ``SLORecommendation`` model ``status`` column.
    """

    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"


class ReviewDecision(str, PyEnum):
    """
    Enumeration of human review decision types.

    Attributes
    ----------
    APPROVE : str
        Reviewer approved the recommendation as-is.
    REJECT : str
        Reviewer rejected the recommendation.
    MODIFY : str
        Reviewer approved a modified version of the recommendation.

    Notes
    -----
    Used in the ``HumanReview`` model ``decision`` column. When ``MODIFY``
    is selected, the ``modified_availability`` and ``modified_latency_p99_ms``
    columns are populated with the reviewer's adjusted values.
    """

    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"


class Service(Base):
    """
    ORM model representing a microservice in the dependency graph.

    Attributes
    ----------
    id : UUID
        Auto-generated UUID v4 primary key.
    name : str
        Unique service identifier used in dependency graph edges.
    display_name : str
        Human-readable service name for UI display.
    description : str or None
        Optional free-text description.
    tier : ServiceTier
        Service criticality tier.
    team : str or None
        Owning team name.
    namespace : str
        Kubernetes namespace. Defaults to ``"default"``.
    region : str
        Deployment region. Defaults to ``"us-east-1"``.
    pagerank_score : float
        Cached PageRank score from the dependency analysis engine.
    betweenness_centrality : float
        Cached betweenness centrality from the dependency analysis engine.
    fan_in : int
        Number of incoming dependency edges.
    fan_out : int
        Number of outgoing dependency edges.
    metadata_ : dict
        Arbitrary metadata stored as JSONB.
    created_at : datetime
        Row creation timestamp set by the database server.
    updated_at : datetime
        Row update timestamp set by the database server.

    Notes
    -----
    ``pagerank_score``, ``betweenness_centrality``, ``fan_in``, and ``fan_out``
    are computed by the dependency math engine and cached here for fast API
    retrieval without re-running the graph analysis.
    """

    __tablename__ = "services"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    tier: Mapped[ServiceTier] = mapped_column(Enum(ServiceTier), nullable=False, default=ServiceTier.MEDIUM)
    team: Mapped[str | None] = mapped_column(String(255))
    namespace: Mapped[str] = mapped_column(String(255), default="default")
    region: Mapped[str] = mapped_column(String(100), default="us-east-1")

    pagerank_score: Mapped[float] = mapped_column(Float, default=0.0)
    betweenness_centrality: Mapped[float] = mapped_column(Float, default=0.0)
    fan_in: Mapped[int] = mapped_column(Integer, default=0)
    fan_out: Mapped[int] = mapped_column(Integer, default=0)

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    outgoing_deps: Mapped[list[ServiceDependency]] = relationship("ServiceDependency", foreign_keys="ServiceDependency.source_id", back_populates="source")
    incoming_deps: Mapped[list[ServiceDependency]] = relationship("ServiceDependency", foreign_keys="ServiceDependency.target_id", back_populates="target")
    metrics: Mapped[list[ServiceMetric]] = relationship("ServiceMetric", back_populates="service")
    slos: Mapped[list[SLO]] = relationship("SLO", back_populates="service")
    recommendations: Mapped[list[SLORecommendation]] = relationship("SLORecommendation", back_populates="service")


class ServiceDependency(Base):
    """
    ORM model representing a directed dependency edge between two services.

    Attributes
    ----------
    id : UUID
        Auto-generated UUID v4 primary key.
    source_id : UUID
        Foreign key to the dependent (calling) service.
    target_id : UUID
        Foreign key to the dependency (called) service.
    dep_type : DependencyType
        Type of dependency relationship.
    weight : float
        Edge weight representing dependency criticality between 0 and 1.
    is_critical_path : bool
        Whether this edge lies on the critical latency path.
    created_at : datetime
        Row creation timestamp set by the database server.

    Notes
    -----
    A unique constraint on ``(source_id, target_id)`` prevents duplicate edges.
    ``weight`` and ``is_critical_path`` are populated by the dependency analysis
    math engine after PageRank and critical path computation.
    """

    __tablename__ = "service_dependencies"
    __table_args__ = (UniqueConstraint("source_id", "target_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("services.id", ondelete="CASCADE"))
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("services.id", ondelete="CASCADE"))
    dep_type: Mapped[DependencyType] = mapped_column(Enum(DependencyType), nullable=False, default=DependencyType.SYNCHRONOUS)

    weight: Mapped[float] = mapped_column(Float, default=1.0)
    is_critical_path: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    source: Mapped[Service] = relationship("Service", foreign_keys=[source_id], back_populates="outgoing_deps")
    target: Mapped[Service] = relationship("Service", foreign_keys=[target_id], back_populates="incoming_deps")


class ServiceMetric(Base):
    """
    Time-bucketed metrics snapshot compatible with TimescaleDB hypertables.

    Attributes
    ----------
    id : UUID
        Auto-generated UUID v4 primary key.
    service_id : UUID
        Foreign key to the parent service.
    bucket_ts : datetime
        Start timestamp of the time bucket (default 1-hour buckets).
    p50_latency_ms : float or None
        50th percentile latency in milliseconds.
    p90_latency_ms : float or None
        90th percentile latency in milliseconds.
    p95_latency_ms : float or None
        95th percentile latency in milliseconds.
    p99_latency_ms : float or None
        99th percentile latency in milliseconds.
    p999_latency_ms : float or None
        99.9th percentile latency in milliseconds.
    request_count : int
        Total number of requests in the bucket.
    error_count : int
        Total number of errors in the bucket.
    error_rate : float
        Error rate as a fraction between 0 and 1.
    availability : float
        Availability as a fraction between 0 and 1.
    cpu_utilisation : float or None
        CPU utilisation between 0 and 1.
    memory_utilisation : float or None
        Memory utilisation between 0 and 1.
    smoothed_p99_ms : float or None
        Kalman-filtered p99 latency written by the math engine.
    smoothed_availability : float or None
        Kalman-filtered availability written by the math engine.

    Notes
    -----
    The composite index on ``(service_id, bucket_ts)`` optimises time-range
    queries for the metrics analysis agent. Kalman-smoothed values are
    back-filled by the metrics processing background task.
    """

    __tablename__ = "service_metrics"
    __table_args__ = (
        Index("ix_service_metrics_service_ts", "service_id", "bucket_ts"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("services.id", ondelete="CASCADE"))
    bucket_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    p50_latency_ms: Mapped[float | None] = mapped_column(Float)
    p90_latency_ms: Mapped[float | None] = mapped_column(Float)
    p95_latency_ms: Mapped[float | None] = mapped_column(Float)
    p99_latency_ms: Mapped[float | None] = mapped_column(Float)
    p999_latency_ms: Mapped[float | None] = mapped_column(Float)

    request_count: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    error_rate: Mapped[float] = mapped_column(Float, default=0.0)

    availability: Mapped[float] = mapped_column(Float, default=1.0)

    cpu_utilisation: Mapped[float | None] = mapped_column(Float)
    memory_utilisation: Mapped[float | None] = mapped_column(Float)

    smoothed_p99_ms: Mapped[float | None] = mapped_column(Float)
    smoothed_availability: Mapped[float | None] = mapped_column(Float)

    service: Mapped[Service] = relationship("Service", back_populates="metrics")


class SLO(Base):
    """
    ORM model representing a service level objective configuration.

    Attributes
    ----------
    id : UUID
        Auto-generated UUID v4 primary key.
    service_id : UUID
        Foreign key to the owning service.
    name : str
        Human-readable SLO name.
    status : SLOStatus
        Lifecycle state of this SLO.
    availability_target : float or None
        Target availability fraction between 0 and 1.
    latency_percentile : float or None
        Percentile for the latency target, e.g. 0.99 for p99.
    latency_target_ms : float or None
        Target latency at the specified percentile in milliseconds.
    error_rate_target : float or None
        Maximum acceptable error rate fraction between 0 and 1.
    window_days : int
        Error budget rolling window in days. Defaults to 30.
    error_budget_remaining : float or None
        Remaining error budget fraction, updated by the background job.
    burn_rate_1h : float or None
        1-hour burn rate updated by the background metrics job.
    burn_rate_6h : float or None
        6-hour burn rate updated by the background metrics job.
    version : int
        Version counter incremented on each SLO update.
    created_at : datetime
        Row creation timestamp.
    updated_at : datetime
        Row update timestamp.

    Notes
    -----
    ``error_budget_remaining``, ``burn_rate_1h``, and ``burn_rate_6h`` are
    updated by a background job that runs the metrics analysis tools on the
    latest observations.
    """

    __tablename__ = "slos"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("services.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[SLOStatus] = mapped_column(Enum(SLOStatus), default=SLOStatus.ACTIVE)

    availability_target: Mapped[float | None] = mapped_column(Float)
    latency_percentile: Mapped[float | None] = mapped_column(Float)
    latency_target_ms: Mapped[float | None] = mapped_column(Float)
    error_rate_target: Mapped[float | None] = mapped_column(Float)

    window_days: Mapped[int] = mapped_column(Integer, default=30)
    error_budget_remaining: Mapped[float | None] = mapped_column(Float)
    burn_rate_1h: Mapped[float | None] = mapped_column(Float)
    burn_rate_6h: Mapped[float | None] = mapped_column(Float)

    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    service: Mapped[Service] = relationship("Service", back_populates="slos")


class SLORecommendation(Base):
    """
    ORM model representing a generated SLO recommendation awaiting review.

    Attributes
    ----------
    id : UUID
        Auto-generated UUID v4 primary key.
    service_id : UUID
        Foreign key to the target service.
    status : RecommendationStatus
        Review status of the recommendation.
    recommended_availability : float or None
        Recommended availability SLO value.
    recommended_latency_p99_ms : float or None
        Recommended p99 latency SLO in milliseconds.
    recommended_error_rate : float or None
        Recommended error rate complement.
    availability_posterior_mean : float or None
        Bayesian posterior mean used to generate the recommendation.
    availability_posterior_std : float or None
        Bayesian posterior standard deviation.
    availability_credible_lower : float or None
        Lower bound of the 95% credible interval.
    availability_credible_upper : float or None
        Upper bound of the 95% credible interval.
    confidence_score : float
        Confidence score between 0 and 1 from the confidence scorer.
    reasoning : str or None
        Engineer-facing explanation of the recommendation.
    data_sources : list
        List of data source identifiers used.
    math_details : dict
        Raw mathematical computation details.
    upstream_impact : dict
        Estimated impact on upstream services.
    downstream_impact : dict
        Estimated impact on downstream services.
    requires_human_review : bool
        Whether HITL review is required before applying.
    review_deadline : datetime or None
        Optional deadline for completing the human review.
    created_at : datetime
        Row creation timestamp.

    Notes
    -----
    Recommendations in ``PENDING_REVIEW`` status are exposed through the
    ``/api/v1/reviews`` endpoint for HITL approval or rejection.
    """

    __tablename__ = "slo_recommendations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("services.id", ondelete="CASCADE"))
    status: Mapped[RecommendationStatus] = mapped_column(Enum(RecommendationStatus), default=RecommendationStatus.PENDING_REVIEW)

    recommended_availability: Mapped[float | None] = mapped_column(Float)
    recommended_latency_p99_ms: Mapped[float | None] = mapped_column(Float)
    recommended_error_rate: Mapped[float | None] = mapped_column(Float)

    availability_posterior_mean: Mapped[float | None] = mapped_column(Float)
    availability_posterior_std: Mapped[float | None] = mapped_column(Float)
    availability_credible_lower: Mapped[float | None] = mapped_column(Float)
    availability_credible_upper: Mapped[float | None] = mapped_column(Float)

    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    reasoning: Mapped[str | None] = mapped_column(Text)
    data_sources: Mapped[list[str]] = mapped_column(JSON, default=list)
    math_details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    upstream_impact: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    downstream_impact: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    requires_human_review: Mapped[bool] = mapped_column(Boolean, default=False)
    review_deadline: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    service: Mapped[Service] = relationship("Service", back_populates="recommendations")
    reviews: Mapped[list[HumanReview]] = relationship("HumanReview", back_populates="recommendation")


class HumanReview(Base):
    """
    ORM model representing an SRE engineer's review decision on a recommendation.

    Attributes
    ----------
    id : UUID
        Auto-generated UUID v4 primary key.
    recommendation_id : UUID
        Foreign key to the reviewed recommendation.
    reviewer : str
        Identifier of the reviewing engineer.
    decision : ReviewDecision
        The reviewer's decision (approve, reject, or modify).
    comment : str or None
        Optional free-text reviewer comment.
    modified_availability : float or None
        Reviewer-adjusted availability when decision is MODIFY.
    modified_latency_p99_ms : float or None
        Reviewer-adjusted p99 latency when decision is MODIFY.
    created_at : datetime
        Review submission timestamp.

    Notes
    -----
    When ``decision=MODIFY``, both ``modified_availability`` and
    ``modified_latency_p99_ms`` should be populated with the adjusted values.
    The review submission triggers a state update on the parent recommendation.
    """

    __tablename__ = "human_reviews"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recommendation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("slo_recommendations.id", ondelete="CASCADE"))
    reviewer: Mapped[str] = mapped_column(String(255), nullable=False)
    decision: Mapped[ReviewDecision] = mapped_column(Enum(ReviewDecision), nullable=False)
    comment: Mapped[str | None] = mapped_column(Text)

    modified_availability: Mapped[float | None] = mapped_column(Float)
    modified_latency_p99_ms: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    recommendation: Mapped[SLORecommendation] = relationship("SLORecommendation", back_populates="reviews")


class AuditLog(Base):
    """
    Immutable audit trail for all recommendation and SLO lifecycle events.

    Attributes
    ----------
    id : UUID
        Auto-generated UUID v4 primary key.
    entity_type : str
        Type of entity being audited, e.g. ``"recommendation"`` or ``"slo"``.
    entity_id : str
        String identifier of the audited entity.
    action : str
        Action performed, e.g. ``"approved"``, ``"applied"``, ``"rejected"``.
    actor : str
        Identifier of the actor performing the action. Defaults to ``"system"``.
    payload : dict
        Arbitrary JSONB payload with event-specific details.
    created_at : datetime
        Event timestamp set by the database server.

    Notes
    -----
    The composite index on ``(entity_type, entity_id)`` optimises queries
    for all audit events related to a specific entity. Audit log rows are
    never updated or deleted.
    """

    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_logs_entity", "entity_type", "entity_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_type: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(255), nullable=False)
    action: Mapped[str] = mapped_column(String(255), nullable=False)
    actor: Mapped[str] = mapped_column(String(255), default="system")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
