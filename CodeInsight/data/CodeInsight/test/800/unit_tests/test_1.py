var0 = [
        (2023, 5, 10, b'SYM1', b'ACT1', 100),
        (2022, 12, 1, b'SYM2', b'ACT2', 200),
        (2022, 6, 15, b'SYM3', b'ACT3', 50),
        (2023, 2, 28, b'SYM4', b'ACT4', 300)
    ]
col0 = ['year', 'month', 'day']
data0 = [('year', '<i4'), ('month', '<i4'), ('day', '<i4'), ('symbol', '|S8'), ('action', '|S4'), ('value', '<i4')]
expected_result =  np.array([ (2022, 6, 15, b'SYM3', b'ACT3', 50), (2022, 12, 1, b'SYM2', b'ACT2', 200), (2023, 2, 28, b'SYM4', b'ACT4', 300), (2023, 5, 10, b'SYM1', b'ACT1', 100) ], dtype=data0)
assert (test(var0, data0, col0)  ==  expected_result).all(), 'Test failed'