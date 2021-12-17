from linear_regress_california import *
given_input = linear_regression(data_set='ADP_dataset.csv')


def test_bad_input(given_input):

    
    given_input.setattr('builtins.input', lambda _: "correct_input")

   
    i = input("enter the file name")
    assert i == "correct_input"