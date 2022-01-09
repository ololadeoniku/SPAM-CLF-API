import uuid
from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model


class SPAMInference(Model):
    __keyspace__ = "spam_inferences"
    uuid = columns.UUID(primary_key=True, default=uuid.uuid1)
    query = columns.Text()
    label = columns.Text()
    confidence_score = columns.Float()
    model_version = columns.Text(default='v1')


class RestAPIUser(Model):
    __keyspace__ = "spam_inferences"
    uuid = columns.UUID(primary_key=True, default=uuid.uuid1)
    email = columns.Text(index=True)
