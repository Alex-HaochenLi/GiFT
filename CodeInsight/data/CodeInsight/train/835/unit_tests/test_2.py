df3 = pd.DataFrame({'grade': ['10.8', '20.4', '30.0']})
var0 ='grade'
expected_result3 = pd.DataFrame({'grade': [10, 20, 30]})
result3 = test(df3, var0)
assert result3.equals(expected_result3), 'Test failed'