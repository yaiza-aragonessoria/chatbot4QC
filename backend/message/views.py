from rest_framework.generics import ListCreateAPIView, RetrieveUpdateDestroyAPIView

from message.models import Message
from message.serializers import MessageSerializer


class ListCreateMessageView(ListCreateAPIView):
    """
    get: Lists all messages of the logged-in in chronological order.

    post: Creates a new Message.
    """
    serializer_class = MessageSerializer

    def get_queryset(self):
        return Message.objects.order_by("updated")

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class RetrieveUpdateDeleteMessageView(RetrieveUpdateDestroyAPIView):
    """
        get: Retrieves a specific Message.
        patch: Updates a specific Message.
        delete: Deletes a Message.
    """
    queryset = Message.objects.all()
    serializer_class = MessageSerializer
    lookup_field = 'id'  # field in the database
    lookup_url_kwarg = 'id_message'  # field in the request
    http_method_names = ['get', 'patch', 'delete']  # disallow put as we don't use it
