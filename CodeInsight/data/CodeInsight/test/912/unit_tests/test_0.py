df0 = pd.DataFrame({'sex': [0, 1, 1, 0, 1, 0, 0]})
col0= 'sex'
var0 = 'Female'
var1 = 'Male'
expected_output = pd.DataFrame({'sex': ['Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Female']})
assert test(df0,col0,var0,var1) .equals(expected_output), 'Test failed'