"""knowledge/models.py  –  문서·레포·임베딩"""

import uuid
from django.db import models
from django.db.models import Q, CheckConstraint
from accounts.models import Organization, User
from pgvector.django import VectorField

class DocType(models.TextChoices):
    POLICY = "policy", "Policy"
    PRODUCT = "product", "Product"
    TECH_MANUAL = "tech_manual", "Tech Manual"


class Document(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    org = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name="documents")
    title = models.CharField(max_length=255)
    doc_type = models.CharField(max_length=20, choices=DocType.choices)
    s3_key = models.TextField(unique=True)
    version = models.CharField(max_length=50, default="v1")
    pinecone_ns = models.CharField(max_length=100)
    uploaded_by = models.ForeignKey(
        User, null=True, blank=True, on_delete=models.SET_NULL, related_name="uploaded_documents"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "documents"
        indexes = [models.Index(fields=["org", "doc_type"], name="idx_docs_org_type")]

    def __str__(self):
        return self.title


class GitRepository(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    org = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name="repositories")
    repo_url = models.TextField(unique=True)
    default_branch = models.CharField(max_length=100, default="main")
    fetched_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "git_repositories"

    def __str__(self):
        return self.repo_url


class CodeFile(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    repo = models.ForeignKey(GitRepository, on_delete=models.CASCADE, related_name="code_files")
    file_path = models.TextField()
    language = models.CharField(max_length=50, blank=True)
    latest_commit = models.CharField(max_length=40, blank=True)
    loc = models.PositiveIntegerField(null=True, blank=True)

    class Meta:
        db_table = "code_files"
        unique_together = ("repo", "file_path")
        indexes = [models.Index(fields=["repo"], name="idx_files_repo")]

    def __str__(self):
        return self.file_path


class EmbedChunk(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(
        Document, null=True, blank=True, on_delete=models.CASCADE, related_name="chunks"
    )
    file = models.ForeignKey(
        CodeFile, null=True, blank=True, on_delete=models.CASCADE, related_name="chunks"
    )
    chunk_index = models.PositiveIntegerField()
    pinecone_id = models.CharField(max_length=100)
    hash = models.CharField(max_length=64, unique=True)

    class Meta:
        db_table = "embed_chunks"
        constraints = [
            CheckConstraint(
                name="embed_chunks_one_fk",
                check=Q(document__isnull=False, file__isnull=True)
                | Q(document__isnull=True, file__isnull=False),
            )
        ]
        indexes = [models.Index(fields=["document", "file"], name="idx_chunks_source")]

    def __str__(self):
        return self.hash
    
class NewsArticle(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    corp = models.CharField(max_length=50, null=False)
    outlier_month = models.DateField(null=False)
    title = models.TextField(null=False)
    publisher = models.CharField(max_length=255)
    published_at = models.DateTimeField(auto_now_add=True)
    content_url = models.TextField(null=False)
    content_embedding = VectorField(dimensions=3072, default=[0.0]*3072)

    class Meta:
        db_table = "news_articles"
    def __str__(self) :
        return self.title
    
class TelcoSubscriberStats(models.Model) :
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    year_month = models.CharField(max_length=7, null=False)
    skt_subscribers = models.BigIntegerField()
    kt_subscribers = models.BigIntegerField()
    lguplus_subscribers = models.BigIntegerField()
    mvno_subscribers = models.BigIntegerField()
    skt_delta = models.BigIntegerField()
    kt_delta = models.BigIntegerField()
    lguplus_delta = models.BigIntegerField()
    mvno_delta = models.BigIntegerField()
    skt_delta_pct = models.DecimalField(max_digits=6, decimal_places=2)
    kt_delta_pct = models.DecimalField(max_digits=6, decimal_places=2)
    lguplus_delta_pct = models.DecimalField(max_digits=6, decimal_places=2)
    mvno_delta_pct = models.DecimalField(max_digits=6, decimal_places=2)
    
    class Meta:
        db_table = "telco_subscriber_stats"


class TelecomCustomers(models.Model) :
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    customer_id = models.CharField(max_length=20, null=False)
    gender = models.CharField(max_length=6)
    senior_citizen = models.BooleanField()
    partner = models.BooleanField()
    dependents = models.BooleanField()
    tenure = models.IntegerField()
    phone_service = models.BooleanField()
    multiple_lines = models.CharField(max_length=20)
    internet_serivce = models.CharField(max_length=20)
    online_security = models.CharField(max_length=20)
    online_backup = models.CharField(max_length=20)
    device_protection = models.CharField(max_length=20)
    tech_support = models.CharField(max_length=20)
    streaming_tv = models.CharField(max_length=20)
    streaming_movies = models.CharField(max_length=20)
    contract = models.CharField(max_length=20)
    paperless_billing = models.BooleanField()
    payment_method = models.CharField(max_length=30)
    monthly_charges = models.DecimalField(max_digits=10, decimal_places=2)
    total_charges = models.DecimalField(max_digits=14, decimal_places=2)
    churn = models.BooleanField()

    class Meta:
        db_table = "telecom_customers"


