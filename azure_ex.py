import azure.quantum
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from azure.quantum import Workspace
from azure.quantum.qiskit import AzureQuantumProvider


connection_string = "SubscriptionId=212d1dac-15dc-4f84-851c-6770cf8695f2;ResourceGroupName=AzureQuantum;WorkspaceName=QuboBenchmar;ApiKey=kNn5YGWI2EaSQ4u4JLbcJAidm9rj48WETWwW7WG1pPuH1LnTWcK11BzrtzHY8t8LL1NJ5U9mwyXOAZQUV433EA;QuantumEndpoint=https://germanywestcentral.quantum.azure.com/;"
workspace = Workspace.from_connection_string(connection_string)
'''
workspace = Workspace(
    resource_id = "/subscriptions/212d1dac-15dc-4f84-851c-6770cf8695f2/resourceGroups/AzureQuantum/providers/Microsoft.Quantum/Workspaces/QuboBenchmar", # Add the resourceID of your workspace
    location = "germanywestcentral" # Add the location of your workspace (for example "westus")
    )
'''

provider = AzureQuantumProvider(workspace)



print("This workspace's targets:")
for backend in provider.backends():
    print("- " + backend.name())



# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(3, 3)
circuit.name = "Qiskit Sample - 3-qubit GHZ circuit"
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)
circuit.measure([0,1,2], [0, 1, 2])

# Print out the circuit
print(circuit.draw())



simulator_backend = provider.get_backend("ionq.simulator")



from qiskit import transpile
circuit = transpile(circuit, simulator_backend, optimization_level=3)



job = simulator_backend.run(circuit, shots=100)
job_id = job.id()
print("Job id", job_id)



result = job.result()
print(result)



counts = {format(n, "03b"): 0 for n in range(8)}
counts.update(result.get_counts(circuit))
print(counts)
plot_histogram(counts)





backend = provider.get_backend("ionq.qpu")
#cost = backend.estimate_cost(circuit, shots=1024)

#print(f"Estimated cost: {cost.estimated_total}")





qpu_backend = provider.get_backend("ionq.qpu") # "quantinuum.qpu.h1-1" "rigetti.qpu.ankaa-2"

# Submit the circuit to run on Azure Quantum
job = qpu_backend.run(circuit, shots=1024)
job_id = job.id()
print("Job id", job_id)

# Get the job results (this method waits for the Job to complete):
result = job.result()
print(result)
counts = {format(n, "03b"): 0 for n in range(8)}
counts.update(result.get_counts(circuit))
print(counts)
plot_histogram(counts)
