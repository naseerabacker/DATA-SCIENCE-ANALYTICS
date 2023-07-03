from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train the classifier
clf = KNeighborsClassifier()
clf.fit(X, y)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get feature inputs from the form
        SL = float(request.form['SL'])
        SW = float(request.form['SW'])
        PL = float(request.form['PL'])
        PW = float(request.form['PW'])

        # Make a prediction using the classifier
        input_features = [[SL, SW, PL, PW]]
        prediction = clf.predict(input_features)

        # Map the predicted class label to the species name
        species = iris.target_names[prediction[0]]

        return render_template("C:/Users/USER/.spyder-py3/index.html",prediction=species)
    else:
        return render_template("C:/Users/USER/.spyder-py3/index.html")
#

if __name__ == '__main__':
    app.run(debug=True)

