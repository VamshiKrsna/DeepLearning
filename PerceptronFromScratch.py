import numpy as np

def unit_step_func(x):
    return np.where(x > 0, 1, 0)

class Perceptron:

    def __init__(self,learning_rate = 0.01,n_iters = 100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None

    def fit(self,x,y):
        n_samples,n_features = x.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0,1,0)

        #Learn Weights:
        for i in range(self.n_iters):
            for index,x_i in enumerate(x):
                linear_output = np.dot(x_i,self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y_[index]-y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self,x):
        linear_output = np.dot(x,self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return  y_predicted


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true,y_preds):
        accuracy = np.sum(y_true == y_preds) / len(y_true)
        return accuracy

    X,y = datasets.make_blobs(
        n_samples=150,n_features=2,centers=2,cluster_std=1.05,random_state=101
    )

    X_train, X_test, y_train, y_test = train_test_split(X,y)

    p = Perceptron(learning_rate=0.01,n_iters=1000)
    p.fit(X_train,y_train)
    preds = p.predict(X_test)


    print("Perceptron Accuracy : ",accuracy(preds,y_test))

    # Plotting Decision Boundary
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(X_train[:,0],X_train[:,1],marker = "o",c = y_train)

    x0_1 = np.amin(X_train[:,0])
    x0_2 = np.amax(X_train[:,0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias)/p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias)/p.weights[1]

    ax.plot([x0_1,x0_2],[x1_1,x1_2],"k")

    ymin = np.amin(X_train[:,1])
    ymax = np.amax(X_train[:,1])
    ax.set_ylim([ymin-3,ymax+3])

    plt.show()