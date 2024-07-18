import nnet
import neural


def test_linear():
    try:
        linear_layer = neural.Linear(in_features=3, out_features=2, is_bias=True)
        print("Linear layer created successfully:", linear_layer)
    except Exception as e:
        print("Error creating Linear layer:", str(e))

test_linear()
