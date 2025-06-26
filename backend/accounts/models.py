"""accounts/models.py  –  조직·사용자"""

import uuid
from django.db import models
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager
from django.utils.text import slugify



class UserRole(models.TextChoices):
    ADMIN = "admin", "Admin"
    ENGINEER = "engineer", "Engineer"
    ANALYST = "analyst", "Analyst"
    GUEST = "guest", "Guest"


class Organization(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "organizations"
        ordering = ["name"]

    def __str__(self):
        return self.name


class UserManager(BaseUserManager):
    def create_user(self, email: str, password: str | None = None, **extra):
        if not email:
            raise ValueError("Email is required")
        user = self.model(email=self.normalize_email(email), **extra)
        user.set_password(password)
        user.save()
        return user

    def create_superuser(self, email: str, password: str | None = None, **extra):
        extra.setdefault("role", UserRole.ADMIN)
        extra.setdefault("is_staff", True)
        extra.setdefault("is_superuser", True)
        
        # Organization이 제공되지 않은 경우 기본 Organization 생성 또는 사용
        if 'org' not in extra:
            # 기본 조직이 있는지 확인
            default_org, created = Organization.objects.get_or_create(
                name="Default Organization"
            )
            extra["org"] = default_org
            
        return self.create_user(email, password, **extra)


def get_profile_image_path(instance, filename):
    """
    Dynamically generates the upload path for a user's profile image.
    Saves images to 'profile_images/<user_name_slug>/<filename>'.
    If the user's name is blank, it uses the user's UUID as the folder name.
    """
    if instance.name:
        folder_name = slugify(instance.name, allow_unicode=True)
    else:
        folder_name = str(instance.id)
    return f'profile_images/{folder_name}/{filename}'


class User(AbstractBaseUser, PermissionsMixin):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    org = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name="users")
    email = models.EmailField(unique=True)
    name = models.CharField(max_length=100, blank=True)
    role = models.CharField(max_length=20, choices=UserRole.choices, default=UserRole.GUEST)
    profile_image = models.ImageField(upload_to=get_profile_image_path, null=True, blank=True, verbose_name="Profile Image")
    created_at = models.DateTimeField(auto_now_add=True)
    last_login = models.DateTimeField(null=True, blank=True)

    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    objects = UserManager()
    USERNAME_FIELD = "email"

    class Meta:
        db_table = "users"
        ordering = ["email"]

    def __str__(self):
        return self.email
