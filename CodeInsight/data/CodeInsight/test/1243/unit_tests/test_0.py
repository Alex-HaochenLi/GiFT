var0 = pd.DataFrame({'A': [0, 0, 0], 'B': [1, 2, 3]})
expected_result =  pd.DataFrame({'B': [1, 2, 3]})
result = test(var0)
assert result.equals(expected_result), 'Test failed'