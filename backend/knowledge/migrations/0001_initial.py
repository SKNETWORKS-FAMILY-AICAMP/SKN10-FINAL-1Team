from django.db import migrations, models
import uuid
from pgvector.django import VectorField

class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ('accounts', '0001_initial'),
    ]

    def install_vector_extension(apps, schema_editor):
        if not schema_editor.connection.vendor.startswith('postgresql'):
            return
        schema_editor.execute('CREATE EXTENSION IF NOT EXISTS vector')

    def reverse_vector_extension(apps, schema_editor):
        if not schema_editor.connection.vendor.startswith('postgresql'):
            return
        schema_editor.execute('DROP EXTENSION IF EXISTS vector')

    operations = [
        migrations.RunPython(
            install_vector_extension,
            reverse_vector_extension
        ),
        migrations.CreateModel(
            name='Document',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('title', models.CharField(max_length=255)),
                ('doc_type', models.CharField(choices=[('policy', 'Policy'), ('product', 'Product'), ('tech_manual', 'Tech Manual')], max_length=20)),
                ('s3_key', models.TextField(unique=True)),
                ('version', models.CharField(default='v1', max_length=50)),
                ('pinecone_ns', models.CharField(max_length=100)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('org', models.ForeignKey(on_delete=models.deletion.CASCADE, related_name='documents', to='accounts.organization')),
                ('uploaded_by', models.ForeignKey(blank=True, null=True, on_delete=models.deletion.SET_NULL, related_name='uploaded_documents', to='accounts.user')),
            ],
            options={
                'db_table': 'documents',
            },
        ),
        migrations.CreateModel(
            name='GitRepository',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('repo_url', models.TextField(unique=True)),
                ('default_branch', models.CharField(default='main', max_length=100)),
                ('fetched_at', models.DateTimeField(blank=True, null=True)),
                ('org', models.ForeignKey(on_delete=models.deletion.CASCADE, related_name='repositories', to='accounts.organization')),
            ],
            options={
                'db_table': 'git_repositories',
            },
        ),
        migrations.CreateModel(
            name='NewsArticle',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('corp', models.CharField(max_length=50)),
                ('outlier_month', models.DateField()),
                ('title', models.TextField()),
                ('publisher', models.CharField(max_length=255)),
                ('published_at', models.DateTimeField(auto_now_add=True)),
                ('content_url', models.TextField()),
                ('content_embedding', VectorField(dimensions=3072, null=True)),
            ],
            options={
                'db_table': 'news_articles',
            },
        ),
        migrations.CreateModel(
            name='TelcoSubscriberStats',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('year_month', models.CharField(max_length=7)),
                ('skt_subscribers', models.BigIntegerField()),
                ('kt_subscribers', models.BigIntegerField()),
                ('lguplus_subscribers', models.BigIntegerField()),
                ('mvno_subscribers', models.BigIntegerField()),
                ('skt_delta', models.BigIntegerField()),
                ('kt_delta', models.BigIntegerField()),
                ('lguplus_delta', models.BigIntegerField()),
                ('mvno_delta', models.BigIntegerField()),
                ('skt_delta_pct', models.DecimalField(decimal_places=2, max_digits=6)),
                ('kt_delta_pct', models.DecimalField(decimal_places=2, max_digits=6)),
                ('lguplus_delta_pct', models.DecimalField(decimal_places=2, max_digits=6)),
                ('mvno_delta_pct', models.DecimalField(decimal_places=2, max_digits=6)),
            ],
            options={
                'db_table': 'telco_subscriber_stats',
            },
        ),
        migrations.CreateModel(
            name='TelecomCustomers',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('customer_id', models.CharField(max_length=20)),
                ('gender', models.CharField(max_length=6)),
                ('senior_citizen', models.BooleanField()),
                ('partner', models.BooleanField()),
                ('dependents', models.BooleanField()),
                ('tenure', models.IntegerField()),
                ('phone_service', models.BooleanField()),
                ('multiple_lines', models.CharField(max_length=20)),
                ('internet_serivce', models.CharField(max_length=20)),
                ('online_security', models.CharField(max_length=20)),
                ('online_backup', models.CharField(max_length=20)),
                ('device_protection', models.CharField(max_length=20)),
                ('tech_support', models.CharField(max_length=20)),
                ('streaming_tv', models.CharField(max_length=20)),
                ('streaming_movies', models.CharField(max_length=20)),
                ('contract', models.CharField(max_length=20)),
                ('paperless_billing', models.BooleanField()),
                ('payment_method', models.CharField(max_length=30)),
                ('monthly_charges', models.DecimalField(decimal_places=2, max_digits=10)),
                ('total_charges', models.DecimalField(decimal_places=2, max_digits=14)),
                ('churn', models.BooleanField()),
            ],
            options={
                'db_table': 'telecom_customers',
            },
        ),
        migrations.CreateModel(
            name='CodeFile',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('file_path', models.TextField()),
                ('language', models.CharField(blank=True, max_length=50)),
                ('latest_commit', models.CharField(blank=True, max_length=40)),
                ('loc', models.PositiveIntegerField(blank=True, null=True)),
                ('repo', models.ForeignKey(on_delete=models.deletion.CASCADE, related_name='code_files', to='knowledge.gitrepository')),
            ],
            options={
                'db_table': 'code_files',
                'unique_together': {('repo', 'file_path')},
            },
        ),
        migrations.CreateModel(
            name='EmbedChunk',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('chunk_index', models.PositiveIntegerField()),
                ('pinecone_id', models.CharField(max_length=100)),
                ('hash', models.CharField(max_length=64, unique=True)),
                ('document', models.ForeignKey(blank=True, null=True, on_delete=models.deletion.CASCADE, related_name='chunks', to='knowledge.document')),
                ('file', models.ForeignKey(blank=True, null=True, on_delete=models.deletion.CASCADE, related_name='chunks', to='knowledge.codefile')),
            ],
            options={
                'db_table': 'embed_chunks',
            },
        ),
        migrations.AddIndex(
            model_name='document',
            index=models.Index(fields=['org', 'doc_type'], name='idx_docs_org_type'),
        ),
        migrations.AddIndex(
            model_name='codefile',
            index=models.Index(fields=['repo'], name='idx_files_repo'),
        ),
        migrations.AddIndex(
            model_name='embedchunk',
            index=models.Index(fields=['document', 'file'], name='idx_chunks_source'),
        ),
        migrations.AddConstraint(
            model_name='embedchunk',
            constraint=models.CheckConstraint(check=models.Q(('document__isnull', False, 'file__isnull', True), ('document__isnull', True, 'file__isnull', False), _connector='OR'), name='embed_chunks_one_fk'),
        ),
    ]
