lst0 = np.array([10,20,30,40,50])
lst1 = np.array([0.2, 0.1, 0.3, 0.1, 0.3])
expected_output = np.sqrt(np.average((lst0-np.average(lst0, weights=lst1))**2, weights=lst1))
assert test(lst0, lst1) == expected_output, 'Test failed'