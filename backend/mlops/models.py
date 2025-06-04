"""mlops/models.py  –  분석 결과 & 모델 레지스트리"""

import uuid
from django.db import models
from accounts.models import User


class ResultType(models.TextChoices):
    CHURN_PRED = "churn_pred", "Churn Prediction"
    VIZ_IMAGE = "viz_image", "Visualization Image"
    TS_FORECAST = "timeseries_forecast", "Time-series Forecast"


class AnalyticsResult(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="analytics_results")
    result_type = models.CharField(max_length=30, choices=ResultType.choices)
    s3_key = models.TextField()
    meta = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "analytics_results"
        indexes = [models.Index(fields=["user", "result_type"], name="idx_analytics_user_type")]

    def __str__(self):
        return f"{self.result_type} - {self.id}"


class ModelStage(models.TextChoices):
    STAGING = "staging", "Staging"
    PRODUCTION = "production", "Production"
    ARCHIVED = "archived", "Archived"


class ModelArtifact(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=120)
    version = models.CharField(max_length=50, default="v1")
    s3_key = models.TextField()
    stage = models.CharField(max_length=20, choices=ModelStage.choices, default=ModelStage.STAGING)
    metrics = models.JSONField(null=True, blank=True)
    created_by = models.ForeignKey(
        User, null=True, on_delete=models.SET_NULL, related_name="model_artifacts"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "model_artifacts"
        unique_together = ("name", "version")
        indexes = [
            models.Index(fields=["name", "stage"], name="idx_model_stage"),
            models.Index(fields=["created_at"], name="idx_model_created"),
        ]
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.version}) - {self.stage}"
