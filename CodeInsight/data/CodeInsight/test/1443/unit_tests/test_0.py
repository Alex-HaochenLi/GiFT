df = pd.DataFrame({'date_column': ['2023-09-13 12:45:30', '2023-09-14 01:23:45']})
df['date_column'] = pd.to_datetime(df['date_column'])
# Test 1
df_test = df.copy()
expected_result =  df.copy()
expected_result['date_column'] = expected_result['date_column'].dt.strftime('%Y-%m-%d')
result = test(df_test, 'date_column', '%Y-%m-%d')
assert result.equals(expected_result), 'Test failed'