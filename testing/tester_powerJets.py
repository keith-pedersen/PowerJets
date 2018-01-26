import powerJets as pJets
import numpy

modeler = pJets.NjetModel()
params = [0, 0.707, 0, -0.5, 1e-3, -0.707, 0, -0.5, 1e-3, 0, 0, 1, 1e-3]

print(modeler(params, 16, 1e5))
