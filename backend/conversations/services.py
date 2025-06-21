from langgraph_sdk import get_client
from django.conf import settings

client = get_client(url=settings.LANGGRAPH_API_URL,api_key=settings.LANGGRAPH_API_KEY)
