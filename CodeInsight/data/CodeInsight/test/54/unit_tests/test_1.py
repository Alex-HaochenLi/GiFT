str0 = "x+y+z"
var0= ",+"
var1 = 1
expected_output = "x+y,+z"
assert test(str0, var0, var1) ==expected_output, 'Test failed'