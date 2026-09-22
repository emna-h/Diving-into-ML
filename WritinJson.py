#------------------------------------------------------------
"""#1. Writing Python Data to a JSON File (json.dump)"""
#--------------------------------------------------------------
import json

patient_info = {
    "patient_id": 101,
    "is_cancerous": False, # Python False converts to JSON true/false
    "features": [-0.237, -0.497, 0.613]
}

# Save Python dictionary into a .json file
with open('patient.json', 'w') as f:
    json.dump(patient_info, f, indent=4) # 'indent=4' formats it neatly
#--------------------------------------------------------------
""""#2. Reading a JSON File into Python (json.load)"""
#-----------------------------------------------------------------
import json

# Read data back into a Python dictionary
with open('patient.json', 'r') as f:
    data = json.load(f)

print(data)
# Output: {'patient_id': 101, 'is_cancerous': False, 'features': [-0.237, -0.497, 0.613]}
print(type(data))
# Output: <class 'dict'>



"""SCENARIO B """
#----------------------------------------------------------
"""1. Convert Python Dict $\rightarrow$ JSON String (json.dumps)"""
#----------------------------------------------------------
import json

patient_dict = {"status": "success", "prediction": 1}

# Convert dictionary to JSON string to send over the network
json_string = json.dumps(patient_dict)

print(json_string)
# Output: '{"status": "success", "prediction": 1}' (a raw text string)



#-------------------------------------------------------------
"""2. Convert JSON String $\rightarrow$ Python Dict (json.loads)"""
#-------------------------------------------------------------
import json

# Incoming raw text received from an API request
incoming_data = '{"patient_id": 101, "is_cancerous": false}'

# Parse JSON string into a usable Python dictionary
patient_dict = json.loads(incoming_data)

print(patient_dict["patient_id"])
# Output: 101


#-------------------------------------------------------------
"""Pro-Tip for ML Engineers: NumPy arrays (np.array([1, 2, 3])) are not directly serializable by json. 
You must convert them to standard Python lists first using .tolist() before dumping them to JSON:"""
#-------------------------------------------------------------
# Convert numpy array to standard list before JSON conversion
json_data = json.dumps({"features": my_numpy_array.tolist()})