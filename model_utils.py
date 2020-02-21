"""Defines the models that we use and the learning processes."""
import logging
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

MONITOR_SEED = 42

# ---------- Abstract base model class ----------

class Model(ABC):
    """
        Abstract class base of the MLP and linear model.
    """
    def __init__(self):
        self.log = dict()

    @abstractmethod
    def save_weights_in_params(self, params, sess):
        """Saves the weights in dictionary params."""

    @abstractmethod
    def define_model(self, X_ph, n_classes, params):
        """Returns logits, l2_pen."""

    def log_to_params(self, params):
        """Puts the parameters in the log function in the params dict."""
        for k, v in self.log.items():
            params[k] = v

    def accumulate_and_log(self, step, names, values):
        """Accumulate all values under their name in dictionary self.log."""
        s_log = "it %5d: " % step
        self.log["step"] = self.log.get("step", list()) + [step]
        for name, value in zip(names, values):
            self.log[name] = self.log.get(name, list()) + [float(value)]
            s_log += "| %s = %5.4f |" % (name, value)
        logging.info(s_log)

    def fit(self, X, Z, Y, params):
        """
            Learning process that fits the model to X, Z, Y.

            Required parameters in params:
                - "X_test", "Z_test" and "Y_test"
                - "display_step"
                - "type_weight"
                - "std_init"
        """
        assert np.all([x in params for x in [
            "display_step", "type_weight", "std_init", "model_type"]])
        assert params["model_type"] == "LINEAR" or ("hl_size" in params)
        n_strata = np.unique(Z).shape[0]
        n_classes = np.unique(Y).shape[0]

        # Placeholders to feed the data:
        X_ph = tf.placeholder(tf.float32, [None, X.shape[1]])
        Z_ph = tf.placeholder(tf.int32, [None]) # Z in range(10)
        Y_ph = tf.placeholder(tf.int32, [None]) # Y in {0, 1}

        # Model definition:
        logits, l2_pen = self.define_model(X_ph, n_classes, params)

        # Loss definition for each instance:
        softmax = tf.nn.softmax(logits, name="softmax")
        one_hot_Y = tf.one_hot(Y_ph, n_classes)
        cross_entropy = - tf.reduce_sum(one_hot_Y*tf.log(softmax), axis=1)

        # Weight the cost
        if params["type_weight"] == "uniform":
            weight_elems = 1
        elif params["type_weight"] == "prediction":
            assert ("p_y_train" in params) and ("p_y_test" in params)
            one_hot_Y = tf.one_hot(Y_ph, n_classes)
            weight_per_class = tf.constant(
                params["p_y_test"]/params["p_y_train"], dtype=tf.float32)
            # Check the dimensions
            weight_elems = tf.reduce_sum(one_hot_Y*weight_per_class, axis=1)
        elif params["type_weight"] == "stratum":
            assert ("p_z_train" in params) and ("p_z_test" in params)
            one_hot_Z = tf.one_hot(Z_ph, n_strata)
            weight_per_class = tf.constant(
                params["p_z_test"]/params["p_z_train"], dtype=tf.float32)
            # Check the dimensions
            weight_elems = tf.reduce_sum(one_hot_Z*weight_per_class, axis=1)

        cost = (tf.reduce_mean(weight_elems*cross_entropy)
                + params["l2_coeff"]*l2_pen)
        cost_unif = tf.reduce_mean(cross_entropy) + params["l2_coeff"]*l2_pen

        # Compute the classification accuracy
        correct_pred = tf.equal(tf.argmax(softmax, 1), tf.argmax(one_hot_Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Compute the top-5 error
        in_top_k = tf.nn.in_top_k(softmax, tf.argmax(one_hot_Y, 1), 5)
        top_k_accuracy = tf.reduce_mean(tf.cast(in_top_k, tf.float32))

        optimizer = tf.train.AdamOptimizer(
            learning_rate=params["learning_rate"],
            beta1=params["momentum"]).minimize(cost)
        # tf.train.MomentumOptimizer(
        # params["learning_rate"], params["momentum"]).minimize(cost)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            step = 0
            sess.run(init)

            test_dict = dict(zip([X_ph, Z_ph, Y_ph],
                                 [params[x] for x in
                                  ["X_test", "Z_test", "Y_test"]]))

            batch_data_tm = select_random_batch([X, Z, Y],
                                                params["batch_size"],
                                                seed=MONITOR_SEED)
            monitor_dict = dict(zip([X_ph, Z_ph, Y_ph], batch_data_tm))

            while step < params["n_ite"]:
                batch_data = select_random_batch([X, Z, Y],
                                                 params["batch_size"])
                train_dict = dict(zip([X_ph, Z_ph, Y_ph], batch_data))

                # print("monitor: ", batch_data_tm[2][0:10])
                # print("train: ", batch_data[2][0:10])

                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict=train_dict)

                if step % params["display_step"] == 0:
                    cost_train, acc_train, topk_train = sess.run(
                        [cost, accuracy, top_k_accuracy],
                        feed_dict=monitor_dict)
                    cost_test, acc_test, topk_test = sess.run(
                        [cost_unif, accuracy, top_k_accuracy],
                        feed_dict=test_dict)
                    l2_pen_ev = sess.run(l2_pen)

                    list_names = ["l2_pen",
                                  "cost_train", "acc_train", "topk_train",
                                  "cost_test", "acc_test", "topk_test"]
                    list_values = [l2_pen_ev,
                                   cost_train, acc_train, topk_train,
                                   cost_test, acc_test, topk_test]

                    self.accumulate_and_log(step, list_names, list_values)

                step += 1

            # Save the weights in list form to make them savable.
            self.save_weights_in_params(params, sess)
        self.log_to_params(params)

# ---------- Implementations of the base model class ----------

class LinearModel(Model):
    """Interface to a linear model to classify the data."""
    def __init__(self):
        self.type_model = "LINEAR"
        super(LinearModel, self).__init__()
        self.weight = None
        self.bias = None
        self.logits = None
        self.l2_pen = None

    def define_model(self, X_ph, n_classes, params):
        n_features = int(X_ph.shape[1])
        w_init = tf.random_normal([n_features, n_classes],
                                  stddev=params["std_init"])

        self.weight = tf.Variable(w_init, name="weights")
        self.bias = tf.Variable(0., name="bias")

        # If logits are negative and too low, the softmax
        # can be close to zero and then it creates NaN in the result.
        self.logits = tf.matmul(X_ph, self.weight) + self.bias
        self.l2_pen = tf.nn.l2_loss(self.weight)

        return self.logits, self.l2_pen

    def save_weights_in_params(self, params, sess):
        ln = ["final_weights", "final_bias"]
        lt = [self.weight, self.bias]

        for name, tensor in zip(ln, lt):
            params[name] = [float(x) for x in sess.run(tensor).ravel()]

class MLP(Model):
    def __init__(self):
        self.type_model = "MLP"
        super(MLP, self).__init__()
        self.weight_1 = None
        self.weight_2 = None
        self.bias_1 = None
        self.bias_2 = None
        self.logits = None
        self.l2_pen = None

    def define_model(self, X_ph, n_classes, params):
        n_features = int(X_ph.shape[1])
        w1_init = tf.random_normal([n_features, params["hl_size"]],
                                   stddev=params["std_init"])
        w2_init = tf.random_normal([params["hl_size"], n_classes],
                                   stddev=params["std_init"])
        b1_init = tf.random_normal([1, params["hl_size"]],
                                   stddev=params["std_init"])
        b2_init = tf.random_normal([1, n_classes],
                                   stddev=params["std_init"])

        self.weight_1 = tf.Variable(w1_init, name="weights_layer_1")
        self.weight_2 = tf.Variable(w2_init, name="weights_layer_2")

        self.bias_1 = tf.Variable(b1_init, name="bias_layer_1")
        self.bias_2 = tf.Variable(b2_init, name="bias_layer_2")

        val_inter = tf.nn.relu(tf.matmul(X_ph, self.weight_1) + self.bias_1)
        self.logits = tf.matmul(val_inter, self.weight_2) + self.bias_2
        self.l2_pen = (tf.nn.l2_loss(self.weight_1)
                       + tf.nn.l2_loss(self.weight_2))

        return self.logits, self.l2_pen

    def save_weights_in_params(self, params, sess):
        ln = ["f_weights_lay1", "f_bias_lay1", "f_weights_lay2", "f_bias_lay2"]
        lt = [self.weight_1, self.bias_1, self.weight_2, self.bias_2]

        for k, v in self.log.items():
            params[k] = v

        for name, tensor in zip(ln, lt):
            params[name] = [float(x) for x in sess.run(tensor).ravel()]


# ----- Small utils -----

def select_random_batch(elems, size_batch, seed=None):
    """Selects a random batch of size size_batch in the elems."""
    n_tot = elems[0].shape[0]
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()
    inds = np.random.randint(0, n_tot, size_batch)
    if seed is not None:
        np.random.seed()
    return [arr[inds] for arr in elems]
