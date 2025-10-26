'''
import model into azureML for monitoring
'''
from azureml.core import Model
from azureml.core.monitoring import ModelDataCollector

model = Model(workspace=ws, name='churn_model')
data_collector = ModelDataCollector(model, output_name=scored_data)

'''
implement model versioning
'''

from azureml.core import Model

model = Model.register(workspace=ws,
                    model_path=outputs/my_model.pkl,
                    model_name=my_model,
                    tags={type: classification},
                    description='A classification model',
                    version=2)

'''
Load testing, for use in Azure cloud shell code
'''
az load test create --name myloadtest --resource-group RG1 --location eastus2
