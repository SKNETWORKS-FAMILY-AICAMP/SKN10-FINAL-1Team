from rest_framework import serializers
from .models import User, Organization

class OrganizationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Organization
        fields = ['id', 'name']

class UserSerializer(serializers.ModelSerializer):
    org = OrganizationSerializer(read_only=True)
    
    class Meta:
        model = User
        fields = ['id', 'email', 'name', 'org', 'role', 'created_at', 'last_login', 'is_active', 'is_staff']
        read_only_fields = ['id', 'email', 'created_at', 'last_login', 'is_active', 'is_staff']
