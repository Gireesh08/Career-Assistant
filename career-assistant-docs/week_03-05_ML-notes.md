Gireesh's ML Learning Notes — Weeks 3, 4 & 5
For RAG-Powered Personal AI Career Assistant
is_float() Function (Data Cleaning)
Definition: A custom Python function that checks whether a value can be converted to a float — used to identify non-numeric entries in a dataframe column
When to Use: During data cleaning when a column expected to contain numbers has messy string entries like ranges ("1000-1200") or garbage values ("34*")
Why We Chose This: The total_sqft column in the Bangalore dataset had mixed types — some values were clean numbers, others were ranges or garbage. Before any math could be done (like price_per_sqft), every value had to be verifiable as a number. is_float() was written manually to give full control over what counts as valid
Key Parameters: None — takes a single value x
Code Example:
python
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

# Usage — find all non-numeric rows
df3[~df3['total_sqft'].apply(is_float)].head(10)
# ~ means NOT — shows rows where is_float returned False
Explain It Like I'm a Beginner: Like a bouncer at a cricket stadium checking tickets — if the ticket scans fine, you're in (True). If it throws an error, you're out (False). The ~ flips it to show all the people who got rejected!
Interview One-Liner: is_float() uses try/except to check if a value converts to float — applied with ~ to find all non-numeric rows in a column
convert_sqft_to_num() Function
Definition: A custom function that converts sqft values to usable numbers — averages ranges like "1000-1200", converts plain strings to float, and returns None for garbage
When to Use: After identifying non-numeric sqft values — this is the cleaning step that follows is_float()
Why We Chose This: Three types of bad data existed: ranges ("1000-1200"), plain strings ("1200"), and garbage ("34*"). A single function handling all three cases was cleaner than three separate cleaning steps
Key Parameters: None — takes a single value x
Code Example:
python
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None
Explain It Like I'm a Beginner: Like a cricket scorer handling messy scorecards — if runs say "45-60" (a range), take the average (52.5). If it says "45" cleanly, just use it. If it says "DNP" (garbage), mark it as unknown (None)!
Interview One-Liner: Splits on - to detect ranges and average them, falls back to direct float conversion, returns None for unparseable garbage
remove_pps_outliers() — Price Per Sqft Outlier Removal
Definition: Location-wise outlier removal that keeps only houses with price_per_sqft within mean ± 1 standard deviation for each location separately
When to Use: Before ML training when extreme price values could skew the model — done location-wise because Whitefield and Yelahanka have completely different normal price ranges
Why We Chose This (over global outlier removal): Global outlier removal would compare a Whitefield house to a Yelahanka house — unfair because their normal ranges are completely different. Location-wise removal ensures each house is only compared to its own neighbourhood. This was a deliberate design choice in our conversation
Key Parameters: Uses mean and std — no tunable parameters
Code Example:
python
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[
            (subdf.price_per_sqft > (m - st)) &
            (subdf.price_per_sqft <= (m + st))
        ]
        df_out = pd.concat([df_out, reduced_df],
                           ignore_index=True)
    return df_out

df7 = remove_pps_outliers(df6)
Explain It Like I'm a Beginner: Like checking if a cricket player's score is normal for their specific team — a score of 30 is great for a tailender but terrible for an opener. You compare within the team, not across all players in the tournament!
Interview One-Liner: Location-wise outlier removal using mean ± std — removes houses priced suspiciously high or low compared to their own neighbourhood, not the global dataset
remove_bhk_outliers() — BHK Anomaly Removal
Definition: Removes listings where a larger BHK costs less per sqft than a smaller BHK in the same location — logically impossible in real estate
When to Use: After price_per_sqft outlier removal — catches a different type of anomaly that pure statistics wouldn't detect
Why We Chose This (over pure statistical methods): Statistical outlier removal based on mean ± std wouldn't catch a 3BHK priced lower than a 2BHK — because the 3BHK price might be within the statistical range. This requires domain knowledge: a bigger flat must cost more per sqft. This was explicitly discussed as a "smarter than statistics" cleaning step
Key Parameters: count > 5 threshold — only compares if smaller BHK has more than 5 listings (enough data to trust the average)
Code Example:
python
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(
                    exclude_indices,
                    bhk_df[bhk_df.price_per_sqft 
                           stats['mean']].index.values
                )
    return df.drop(exclude_indices, axis='index')
Explain It Like I'm a Beginner: Like finding that a senior cricket player earns less than a junior in the same team — that's impossible and must be a data error. The function finds and removes all such impossible listings!
Interview One-Liner: Uses domain knowledge (bigger BHK must cost more per sqft) to remove logically impossible listings that pure statistics wouldn't catch
predict_price() Function
Definition: A function that builds the correct input row for the trained model by filling sqft/bath/bhk at fixed positions and putting 1 at the right location column
When to Use: At inference time — when a user provides location, sqft, bath, bhk and wants a predicted price
Why We Chose This: The model was trained on a specific column order (sqft at 0, bath at 1, bhk at 2, then 200+ one-hot location columns). The predict function must recreate that exact structure — any mismatch would give wrong predictions
Key Parameters: location (string), sqft (float), bath (int), bhk (int)
Code Example:
python
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))  # blank form — all zeros
    x[0] = sqft   # sqft always at index 0
    x[1] = bath   # bath always at index 1
    x[2] = bhk    # bhk always at index 2
    if loc_index >= 0:
        x[loc_index] = 1  # put 1 at location's column
    return lr_clf.predict([x])[0]
Explain It Like I'm a Beginner: Like filling a job application form — most boxes are for location (one box per city, tick your city). First 3 boxes are always fixed fields. The function fills this form exactly as the model expects, then hands it to the model for a prediction!
Interview One-Liner: Builds a zero-filled input array matching training column structure, fills sqft/bath/bhk at fixed indices, puts 1 at the location column, feeds to model
GridSearchCV for Model Selection
Definition: Exhaustively tries every combination of hyperparameters across multiple models using cross validation — returns the best model and best settings
When to Use: When comparing multiple models with different hyperparameters — automates what would otherwise be manual trial and error
Why We Chose This (over manual tuning): Manual tuning is guesswork and doesn't guarantee finding the best combination. GridSearchCV tries ALL combinations systematically — for 3 models with 4 combos each × 5 folds = 60 training runs automatically
Key Parameters: estimator, param_grid, cv, return_train_score
Code Example:
python
def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {}
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['squared_error', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'],
                          cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores,
                        columns=['model', 'best_score', 'best_params'])
Explain It Like I'm a Beginner: Like a cricket selector who tries every possible batting lineup and bowling combination across 5 practice matches — automatically picks the combo that gave the best average performance across all matches!
Interview One-Liner: GridSearchCV tries every hyperparameter combination with cross validation and returns the best performing model and configuration — Linear Regression won at 81.8% in our Bangalore project
Flask REST API (server.py + util.py Architecture)
Definition: A lightweight Python web framework used to serve ML model predictions through HTTP endpoints — the instructor split logic into server.py (routing) and util.py (ML work)
When to Use: When you want to make your trained ML model accessible through a webpage or API
Why We Chose This (over putting everything in one file): The instructor split server.py and util.py for separation of concerns — server.py is the waiter (handles requests), util.py is the chef (does ML work). If prediction breaks, you look at util.py. If routing breaks, you look at server.py. Easier to debug
Key Parameters: @app.route, methods=['GET'/'POST'], host='0.0.0.0', port
Code Example:
python
# server.py — routing only
from flask import Flask, request, jsonify, send_from_directory
import util, os

app = Flask(__name__)

@app.route('/')
def serve_index():
    return send_from_directory('../Client', 'app.html')

@app.route('/predict_home_price', methods=['GET', 'POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])
    response = jsonify({
        'estimated_price': util.get_estimated_price(
            location, total_sqft, bhk, bath)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    util.load_saved_artifacts()
    app.run(host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)))
Explain It Like I'm a Beginner: Flask is the waiter in a restaurant. server.py takes orders (requests) from customers (webpage). util.py is the chef who actually cooks (loads model, predicts). The waiter never cooks — the chef never talks to customers. Clean separation!
Interview One-Liner: Flask routes HTTP requests to Python functions — server.py handles routing, util.py handles model loading and prediction, deployed on Render with host='0.0.0.0' to accept external connections
util.py — load_saved_artifacts() and Private Variables
Definition: A Python module that loads the trained pickle model and columns.json once at server startup — uses private global variables and getter functions
When to Use: Any Flask ML project where model loading should happen once (not on every request)
Why We Chose This: Loading a model on every prediction request is slow and wasteful. Loading once at startup and storing in a global variable means every prediction uses the already-loaded model — much faster
Key Parameters: global keyword, __ (double underscore) private variables
Code Example:
python
# util.py
import pickle, json, numpy as np

__locations = None    # private — only util.py touches this
__data_columns = None
__model = None

def load_saved_artifacts():
    global __data_columns, __locations, __model

    with open('Model/columns.json', 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
        # first 3 are sqft, bath, bhk — rest are locations!

    if __model is None:
        with open('Model/banglore_home_prices_model.pickle',
                  'rb') as f:
            __model = pickle.load(f)

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return round(__model.predict([x])[0], 2)

def get_location_names():
    return __locations
Explain It Like I'm a Beginner: Like a cricket team's shared whiteboard — load_saved_artifacts() writes the team strategy on it once before the match. Every player (function) reads from the same board during the match — nobody re-writes it every time they need to check!
Interview One-Liner: Double underscore makes variables private to the module, global keyword allows updating them inside functions, load_once pattern prevents expensive model reloading on every prediction
Neural Networks — Architecture and Forward Pass
Definition: Layers of neurons (numbers) connected by weights — data flows forward through layers, each layer transforming the input until a final prediction is made
When to Use: Complex pattern recognition where traditional ML (linear regression, trees) doesn't capture the complexity
Why We Chose This (over Linear Regression for MNIST): Linear Regression outputs a continuous number — not suitable for classifying 10 digit classes. Neural Networks with softmax output give probabilities for each class. Also, pixel relationships are non-linear — neural networks capture this, linear models can't
Key Parameters: units (neurons per layer), activation, input_shape, layers
Code Example:
python
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,),
                       activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
# Input(784) → Hidden(100, ReLU) → Output(10, Softmax)
# Total parameters: 79,510
Explain It Like I'm a Beginner: Like your brain when a cricket ball comes at you — eyes (input layer, 784 pixels) send signals to brain (hidden layer, 100 neurons) which decides what to do (output layer, 10 classes). Each connection between neurons is a weight!
Interview One-Liner: Neural networks are stacked layers of neurons connected by learnable weights — forward pass transforms input through layers to produce predictions, trained by backpropagation
Gradient Descent
Definition: An optimisation algorithm that iteratively updates model weights by moving in the direction that reduces the loss — like rolling a ball downhill to find the valley
When to Use: Training any neural network — happens automatically inside model.fit()
Why We Chose This (Adam over plain SGD): Plain SGD uses the same learning rate for all weights. Adam (Adaptive Moment Estimation) adjusts the learning rate individually for each weight — like a smart student who studies harder on weak topics. Adam was chosen because it converges faster and works well out of the box
Key Parameters: learning_rate (default 0.001 for Adam), optimizer type
Code Example:
python
# Manual implementation (as coded in tutorial)
def gradient_descent(age, affordability, y_true,
                     epochs, loss_threshold):
    w1 = w2 = 1
    bias = 0
    rate = 0.5
    n = len(age)
    for i in range(epochs):
        weighted_sum = w1*age + w2*affordability + bias
        y_predicted = sigmoid_numpy(weighted_sum)
        loss = log_loss(y_true, y_predicted)
        w1d = (1/n)*np.dot(np.transpose(age),
                           (y_predicted - y_true))
        w2d = (1/n)*np.dot(np.transpose(affordability),
                           (y_predicted - y_true))
        bias_d = np.mean(y_predicted - y_true)
        w1 = w1 - rate * w1d
        w2 = w2 - rate * w2d
        bias = bias - rate * bias_d
        if loss <= loss_threshold:
            break
    return w1, w2, bias

# In Keras — Adam handles this automatically
model.compile(optimizer='adam', ...)
Explain It Like I'm a Beginner: Blindfolded on a mountain trying to reach the valley. You feel the slope under your feet (gradient), take a small step downhill (weight update), feel again, step again — repeat until you reach the lowest point (minimum loss). Adam is a smart version that takes bigger steps on steep slopes and smaller steps near the bottom!
Interview One-Liner: Gradient descent moves weights opposite to the gradient to minimise loss — Adam adapts the learning rate per weight, converging faster than plain SGD
Backpropagation
Definition: The algorithm that calculates each weight's contribution to the prediction error by propagating error signals backwards through the network — then gradient descent uses these to update weights
When to Use: Training neural networks — automatic in Keras, but coded manually in the gradient_descent() function in the tutorial
Why We Chose This: There's no alternative — backpropagation is the only computationally feasible way to train deep networks. It efficiently computes all gradients in one backward pass using the chain rule
Key Parameters: Not directly configurable — controlled by optimizer
Code Example:
python
# The w1d calculation IS backpropagation manually coded:
w1d = (1/n) * np.dot(np.transpose(age),
                     (y_predicted - y_true))
# This calculates how much w1 caused the error
# Then gradient descent uses it:
w1 = w1 - rate * w1d  # fix w1 by small amount

# In Keras — happens automatically:
model.fit(X_train, y_train, epochs=15)
# Each epoch: forward → loss → backprop → update
Explain It Like I'm a Beginner: Team loses a match (wrong prediction). Captain goes backwards asking: "Bowlers — your fault? Fix your line! Fielders — your fault? Fix positions! Batsmen — your fault? Fix technique!" Everyone adjusts based on how much they contributed to the loss. Next match — better performance!
Interview One-Liner: Backpropagation propagates prediction error backwards through layers, calculating each weight's gradient — gradient descent then uses these gradients to update weights
Activation Functions (ReLU, Sigmoid, Softmax)
Definition: Mathematical functions applied after each layer's weighted sum — introducing non-linearity so the network can learn complex patterns
When to Use: Always — without activation functions, stacking layers is mathematically equivalent to one layer (just multiplication)
Why We Chose This: ReLU for hidden layers (gradient stays 1 for positive values — no vanishing gradient). Softmax for output layer (all 10 class probabilities sum to exactly 1.0 — clear winner). Sigmoid would cause vanishing gradient in hidden layers
ReLU
python
# if x < 0 → output 0 (killed!)
# if x > 0 → output x (passed through!)
keras.layers.Dense(100, activation='relu')
# Used in: hidden layers
# Why: fast + gradient = 1 for positive → no vanishing gradient!
Softmax
python
# All outputs sum to 1.0 — proper probabilities
keras.layers.Dense(10, activation='softmax')
# Used in: multi-class output layer
# Output: [0.01, 0.02, 0.91, ...] → digit 2 predicted!
Sigmoid
python
# Squishes to 0-1 but independently per neuron
keras.layers.Dense(1, activation='sigmoid')
# Used in: binary classification output only
# NOT for hidden layers — causes vanishing gradient!
Explain It Like I'm a Beginner: ReLU is a cricket selector — negative score means "sit out" (output 0), positive score means "play on" (keep the value). Softmax converts all player scores to percentage chances of being captain — always adds to 100%. Sigmoid converts each player's score independently — doesn't guarantee they all add up!
Interview One-Liner: ReLU (hidden layers) passes positive values unchanged, kills negatives, gradient=1 prevents vanishing. Softmax (output) converts scores to probabilities summing to 1 for multi-class classification
Vanishing Gradient Problem
Definition: When gradients become extremely tiny as they flow backwards through many layers — early layers receive near-zero gradient signals and stop learning entirely
When to Use: Understanding this explains why Sigmoid fails in hidden layers and why ReLU was invented
Why We Chose ReLU over Sigmoid for hidden layers: Sigmoid's maximum gradient is 0.25. After 4 layers: 0.25⁴ = 0.004 — almost zero. Early layers hear almost nothing and stop learning. ReLU's gradient is 1 for positive values — 1⁴ = 1. Signal stays strong all the way back. This was the key insight from our conversation
Key Parameters: Not a model parameter — solved by choosing ReLU activation
Code Example:
python
# CAUSES vanishing gradient — don't use in hidden layers:
keras.layers.Dense(100, activation='sigmoid')  # ❌

# PREVENTS vanishing gradient — use in hidden layers:
keras.layers.Dense(100, activation='relu')     # ✅

# Sigmoid MAX gradient = 0.25
# After 4 layers: 0.25 × 0.25 × 0.25 × 0.25 = 0.004
# Nearly zero — early layers stop learning!

# ReLU gradient = 1 for positive values
# After 4 layers: 1 × 1 × 1 × 1 = 1
# Still strong — ALL layers keep learning!
Explain It Like I'm a Beginner: Whispering a message through 10 players — each player repeats it at 25% volume. By player 1, it's silent! That's Sigmoid. ReLU is like each player repeating at full volume — player 1 hears it clearly!
Interview One-Liner: Sigmoid's max gradient of 0.25 multiplies per layer becoming ~0 in deep networks — ReLU's gradient of 1 stays constant, allowing all layers to keep learning
Log Loss (Binary Cross Entropy)
Definition: A loss function for classification that penalises confident wrong predictions much more severely than uncertain wrong predictions — uses logarithms
When to Use: Classification problems — binary or multi-class. NOT for regression (use MSE instead)
Why We Chose This (over MSE for classification): MSE penalises proportionally to distance — a confident wrong prediction and an uncertain wrong prediction get similar penalties. Log loss penalises confident wrong predictions exponentially — "99% sure and wrong" gets massively penalised vs "55% sure and wrong" — much better signal for classification training
Key Parameters: epsilon (1e-15) — prevents log(0) crash
Code Example:
python
def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    # Clip predictions away from 0 and 1 (prevent log crash)
    y_predicted_new = [max(i, epsilon) for i in y_predicted]
    y_predicted_new = [min(i, 1-epsilon)
                       for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(
        y_true * np.log(y_predicted_new) +
        (1 - y_true) * np.log(1 - y_predicted_new)
    )

# In Keras:
model.compile(loss='sparse_categorical_crossentropy', ...)
# sparse_ prefix because labels are integers (0,1,2...)
# not one-hot encoded ([0,0,1,0,...])
Explain It Like I'm a Beginner: Like a cricket commentator being punished for wrong predictions. "99% sure India wins" and India loses → massive punishment! "55% sure India wins" and India loses → small punishment. The more confident and wrong, the bigger the penalty!
Interview One-Liner: Log loss exponentially penalises confident wrong predictions — epsilon clips predictions to prevent log(0), sparse_ prefix means labels are integers not one-hot encoded
MAE (Mean Absolute Error) — Custom Implementation
Definition: Average of absolute differences between actual and predicted values — tells you on average how many units (Lakhs, digits) off your predictions are
When to Use: Regression evaluation — more interpretable than RMSE because it's in the same units as your target variable
Why We Chose This: The instructor coded MAE manually (not using sklearn) specifically to show what's happening mathematically — abs() prevents positive and negative errors from cancelling each other out
Key Parameters: None — pure calculation
Code Example:
python
def mae(y_true, y_predicted):
    total_error = 0
    for yt, yp in zip(y_true, y_predicted):
        total_error += abs(yt - yp)
        # abs() ensures errors always add up, never cancel!
    print("Total Error:", total_error)
    mae = total_error / len(y_true)
    print("MAE:", mae)
    return mae

# sklearn equivalent (what you'd use in production):
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_true, y_predicted)
Explain It Like I'm a Beginner: You predicted 5 cricket match scores. Errors were 30, -10, 20, -15, 5 runs. Without abs(), they partially cancel (30). With abs(), total error = 80, MAE = 16 runs average error. abs() gives the true picture!
Interview One-Liner: MAE is the average absolute difference between actual and predicted — abs() prevents positive and negative errors from cancelling, giving true average error
Confusion Matrix
Definition: An N×N table comparing actual labels (rows) vs predicted labels (columns) — diagonal = correct predictions, off-diagonal = mistakes
When to Use: Evaluating any classification model — especially multi-class where accuracy alone doesn't show which classes are being confused
Why We Chose This: 97.49% accuracy sounds great, but it doesn't tell you if the model confuses 4s with 9s, or 3s with 8s. Confusion matrix reveals the specific failure patterns
Key Parameters: labels (actual), predictions (predicted), annot=True, fmt='d', cmap='Blues'
Code Example:
python
y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
# argmax finds INDEX of highest probability = predicted digit

cm = tf.math.confusion_matrix(
    labels=y_test,
    predictions=y_predicted_labels
)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix — MNIST Digit Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
Explain It Like I'm a Beginner: Like a cricket scorecard — overall accuracy tells you the final match result, but confusion matrix tells you exactly which shot got the player out, against which bowler, on which delivery. Diagonal = all the correct predictions (dark blue = high count = good!)
Interview One-Liner: Confusion matrix is an NxN grid of actual vs predicted — diagonal shows correct predictions, off-diagonal shows which classes the model confuses, visualised as a heatmap
CNN (Convolutional Neural Network)
Definition: A neural network architecture that uses sliding filters to detect spatial features in images — far more efficient than Dense networks for image data
When to Use: Any image classification task — especially colour images with complex patterns like CIFAR-10
Why We Chose This (over Dense for CIFAR-10): Dense networks flatten the image and lose all spatial information — a pixel at position (5,5) loses its relationship to pixel at (5,6). CNNs preserve spatial structure. Also, Dense networks would need billions of weights for large colour images — CNNs use shared filter weights making them computationally feasible
Key Parameters: filters, kernel_size, activation, pool_size, padding, strides, input_shape
Code Example:
python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

model = Sequential([
    # Conv Block 1 — basic features (edges, lines)
    Conv2D(filters=32, kernel_size=(3,3),
           activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(pool_size=(2,2)),
    # Output: (15, 15, 32)

    # Conv Block 2 — complex features (shapes, textures)
    Conv2D(filters=64, kernel_size=(4,4),
           activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    # Output: (6, 6, 64)

    Flatten(),
    # Output: (2304,)

    Dense(units=34, activation='relu'),
    Dense(units=10, activation='softmax')
])
Explain It Like I'm a Beginner: Like a cricket scout who doesn't look at the whole match at once. Instead, he watches small sections of the field one at a time (sliding filter), notes what he sees in each section (feature map), then combines all his notes to judge the overall player (Dense layer → prediction)!
Interview One-Liner: CNNs use sliding filters to detect spatial features — Conv2D extracts features, MaxPooling reduces size, Flatten bridges to Dense, Softmax gives class probabilities
Conv2D Layer Parameters
Definition: The core convolutional layer — slides learnable filters over the image to produce feature maps
When to Use: First and subsequent layers of any CNN for image processing
Why We Chose 32→64 filter progression: Layer 1 (32 filters) detects simple features — edges, lines, corners. Layer 2 (64 filters) combines simple features into complex patterns — shapes, textures, object parts. More filters needed for complexity. This progression was specifically discussed in our conversation
Key Parameters:
filters — how many different features to detect
kernel_size — size of sliding filter (3×3 or 4×4)
activation — relu for hidden conv layers
input_shape — only on first layer
padding='same' — keeps size constant
padding='valid' — size shrinks (default)
Code Example:
python
# First conv layer — 32 filters, 3×3 kernel
Conv2D(filters=32, kernel_size=(3,3),
       activation='relu', input_shape=(32,32,3))
# Input: (32,32,3) → Output: (30,30,32)
# 32-3+1 = 30 (valid padding reduces by kernel_size-1)

# Second conv layer — 64 filters, 4×4 kernel
Conv2D(filters=64, kernel_size=(4,4), activation='relu')
# Input: (15,15,32) → Output: (12,12,64)
# 15-4+1 = 12
Explain It Like I'm a Beginner: Like having 32 different magnifying glasses, each looking for a different thing (one for horizontal lines, one for vertical, one for curves). Slide all 32 glasses over the image simultaneously — each produces a map showing WHERE it found its feature!
Interview One-Liner: Conv2D slides filters over image producing feature maps — filters=32 means 32 feature detectors, kernel_size=(3,3) means each filter sees a 3×3 pixel area
MaxPooling2D
Definition: Downsampling layer that takes the maximum value from each non-overlapping pooling window — reduces feature map size while retaining the strongest features
When to Use: After every Conv2D layer — reduces computation, prevents overfitting, provides translation invariance
Why We Chose MaxPooling (over AveragePooling): MaxPooling keeps the STRONGEST activation from each region — the most clearly detected feature. AveragePooling would dilute strong signals by averaging with weak ones. For image features, the presence (maximum) matters more than the average
Key Parameters: pool_size (typically 2,2)
Code Example:
python
MaxPooling2D(pool_size=(2,2))
# Takes max from each 2×2 block
# Input: (30,30,32) → Output: (15,15,32)
# Exactly half the spatial dimensions!

# What it does to one 2×2 block:
# [3, 7]  → MaxPool → 7 (keeps maximum!)
# [2, 5]
Explain It Like I'm a Beginner: Like summarising 4 paragraphs into 1 sentence — you keep the most important point (maximum value) from each group of 4 pixels. You throw away the weaker 3 but keep the strongest signal! Result: half the size, same important information!
Interview One-Liner: MaxPooling takes the maximum value from each pooling window — halves spatial dimensions, preserves strongest features, reduces computation and overfitting
Flatten Layer
Definition: Converts 3D feature maps (Height × Width × Filters) into a 1D array — the necessary bridge between convolutional layers and Dense layers
When to Use: Once, after the last Conv/Pooling block and before the first Dense layer
Why We Chose This: Dense layers only accept 1D input. After convolution and pooling you have 3D feature maps. Flatten is the only way to connect them — there's no alternative. No parameters needed
Key Parameters: None
Code Example:
python
Flatten()
# Input: (6, 6, 64)     ← 3D feature maps
# Output: (2304,)        ← 1D array
# 6 × 6 × 64 = 2304 numbers total

# After Flatten → Dense layers process normally:
Dense(34, activation='relu')
Dense(10, activation='softmax')
Explain It Like I'm a Beginner: Like unrolling a 3D ball of yarn into one long straight thread. Same content, different shape. The Dense layer can only read a straight thread — not a ball!
Interview One-Liner: Flatten converts 3D feature maps to 1D by concatenating all values — required bridge between Conv/Pooling layers and Dense layers, total neurons = H × W × Filters
MNIST Dataset and Project Results
Definition: 70,000 greyscale images of handwritten digits (0-9), 28×28 pixels each — the standard beginner deep learning benchmark
When to Use: Learning neural networks with Dense layers — simpler than CIFAR-10, achieves very high accuracy
Why We Chose This: MNIST was chosen as the Week 4 project because it's complex enough to justify a neural network (97%+ accuracy) but simple enough that a 2-layer Dense network works — no CNNs needed. The progression from MNIST (Dense NN) to CIFAR-10 (CNN) was deliberate
Key Parameters: 60,000 training images, 10,000 test images, 10 classes
Code Example:
python
# Load and preprocess
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize
X_train = X_train / 255
X_test = X_test / 255

# Flatten 28×28 → 784 (Dense needs 1D input)
X_train_flat = X_train.reshape(len(X_train), 28*28)
X_test_flat = X_test.reshape(len(X_test), 28*28)

# Model
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,),
                       activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train_flat, y_train,
                    epochs=15, validation_split=0.2)

# Results:
# Training Accuracy:   99.70%
# Test Accuracy:       97.49%  ✅
# Total Parameters:    79,510
Explain It Like I'm a Beginner: MNIST is like a beginner cricket training camp — the drills are straightforward (recognise digits), the ground is small (28×28 pixels), and even a simple 2-player team (2 Dense layers) can score 97.49%!
Interview One-Liner: MNIST: 70k greyscale 28×28 digit images, 2-layer Dense NN (784→100→10), achieved 97.49% test accuracy with Adam optimizer and 15 epochs
CIFAR-10 Dataset
Definition: 60,000 colour images (32×32×3) of 10 object classes — airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks
When to Use: Learning CNNs — more complex than MNIST because colour + real-world objects
Why We Chose This (over other datasets): CIFAR-10 is the natural next step after MNIST — introduces colour (3 channels), larger images, and real-world complexity that forces you to use CNNs instead of Dense networks. Standard Week 5 project in ML curricula
Key Parameters: 50,000 training, 10,000 test, 32×32×3 = 3,072 numbers per image
python
# MNIST vs CIFAR-10 comparison:
# MNIST:    (28, 28)    → 784 numbers  → Dense NN works
# CIFAR-10: (32, 32, 3) → 3,072 numbers → needs CNN!

y_classes = ['airplane', 'automobile', 'bird', 'cat',
             'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def showimage(x, y, index):
    plt.figure(figsize=(15,2))
    plt.imshow(x[index])
    plt.xlabel(y_classes[y[index]])
    # y[index] → label number (e.g. 3)
    # y_classes[3] → 'cat' (human readable!)
Explain It Like I'm a Beginner: CIFAR-10 is like a professional cricket tournament — the game is much harder (colour images, real objects), the ground is bigger (32×32), and you need a specialist team (CNN) instead of generalists (Dense NN)!
Interview One-Liner: CIFAR-10 has 60k colour 32×32×3 images across 10 classes — requires CNN because Dense networks lose spatial information and would need billions of weights
Image Normalization
Definition: Dividing pixel values by 255 to convert from 0-255 range to 0-1 range before feeding to neural network
When to Use: Always — before any neural network training on image data
Why We Chose Dividing by 255: Images are stored as uint8 (8-bit unsigned integer) — values always range 0-255. Dividing by the maximum (255) always gives 0-1. Neural networks learn much faster with small numbers — gradients are more stable, training converges quicker. This applies to BOTH MNIST and CIFAR-10
Key Parameters: None — just divide by 255
Code Example:
python
# Check before normalizing
print("Min:", X_train.min())   # 0
print("Max:", X_train.max())   # 255

X_train = X_train / 255
X_test  = X_test / 255

# Verify
print("Min:", X_train.min())   # 0.0
print("Max:", X_train.max())   # 1.0

# For CNN (CIFAR-10) → keep shape, just normalize:
# X_train.shape stays (50000, 32, 32, 3) — no flattening!

# For Dense NN (MNIST) → normalize then flatten:
X_train_flat = X_train.reshape(len(X_train), 28*28)
Explain It Like I'm a Beginner: Like converting cricket scores from raw runs (0-400) to percentage of target (0-1). Much easier to compare and work with — and your neural network "brain" works the same way!
Interview One-Liner: Dividing by 255 maps uint8 pixel values to 0-1 — neural networks converge faster and more stably with normalised inputs
Doubts, Debugging & Corrections
Doubt 1 — get_dummies returning True/False instead of 0/1

Asked: Why does pd.get_dummies return True/False instead of 0 and 1 like the instructor?
Resolution: Newer pandas versions (1.5+) changed the default dtype to boolean. Fix:

python
pd.get_dummies(df.location, dtype=int)  # forces 0 and 1 ✅

Note: True/False still works mathematically — Python treats True=1, False=0 internally. It's a cosmetic difference for readability.

Doubt 2 — Why x[0]=sqft, x[1]=bath, x[2]=bhk in predict_price

Asked: Why are these specific positions used? Why not some other order?
Resolution: Because the training dataframe X had sqft as column 0, bath as column 1, bhk as column 2 — followed by 200+ one-hot location columns. The predict function must match this exact column order. The blank form np.zeros() represents an empty version of this same structure. If you put sqft at position 1 instead, the model would interpret it as bath — wrong prediction!

Doubt 3 — Double underscore __ variables in util.py

Asked: Why are variables written as __locations instead of just locations?
Resolution: Double underscore signals "private to this module." server.py should not directly access __locations — it should call get_location_names() instead. This is a code organisation convention. Python doesn't strictly enforce it but it signals intent to other developers (and yourself later!).

Doubt 4 — Why global keyword inside load_saved_artifacts()

Asked: Why write global __model inside a function?
Resolution: Without global, Python creates a NEW local variable inside the function that gets destroyed when the function returns. The global __model at the top of the file would stay None forever. With global, Python knows "use the one at the top of the file, don't create a new one." This is what allows get_estimated_price() to use the loaded model later.

Doubt 5 — File modes: r, rb, w, wb

Asked: What does rb mean? What does f mean? Where else is r used?
Resolution:

python
'r'  → read TEXT file  (columns.json is text)
'rb' → read BINARY file (pickle is not plain text!)
'w'  → write TEXT file
'wb' → write BINARY file
'b'  → means binary — pickle files contain encoded data,
       not human-readable characters

as f → f is just a nickname for the opened file
       (convention — could be named anything)
Doubt 6 — (32, 32, 3) array shape confusion

Asked: What does ndarray (32, 32, 3) mean exactly?
Resolution:

32 → image height (32 rows of pixels)
32 → image width (32 columns of pixels)
3  → colour channels: Red, Green, Blue

Total values = 32 × 32 × 3 = 3,072 numbers per image

Compare with MNIST:
(28, 28) → greyscale, no colour channel
28 × 28 = 784 numbers per image
Doubt 7 — normalize parameter removed from sklearn

Asked: GridSearchCV throwing errors with normalize parameter
Resolution: normalize was deprecated and removed in newer sklearn versions. Solution:

python
# Remove normalize from LinearRegression params:
'linear_regression': {
    'model': LinearRegression(),
    'params': {}  # empty — no params to tune!
}
# Also: criterion='mse' renamed to criterion='squared_error'
Doubt 8 — Flask doesn't work in Google Colab

Asked: Can I run Flask server.py in Colab instead of VS Code like the instructor uses PyCharm?
Resolution: No — Colab runs on Google's remote servers. Flask creates a local server on your machine (localhost:5000). Your browser can't reach a Flask server running inside Colab. Solution: use VS Code on laptop for Flask projects. Colab is only for ML model training and Jupyter-style notebooks.

Doubt 9 — What's the difference between Sigmoid and Softmax for output layer

Asked: Instructor used sigmoid but you mentioned softmax — which is correct?
Resolution: Both work but Softmax is more correct for multi-class:

Sigmoid: each of the 10 neurons outputs independently
         → values don't necessarily sum to 1
         → could have neuron 2 = 0.95 AND neuron 7 = 0.89
         → ambiguous!

Softmax: all 10 neurons computed together
         → always sum to exactly 1.0
         → neuron 2 = 0.91, all others share 0.09
         → clear winner! ✅

For 10-class MNIST → Softmax is the correct choice
Doubt 10 — Why argmax is needed on model.predict() output

Asked: Why do we need np.argmax(i) on the predictions?
Resolution: model.predict() returns probabilities for each class:

python
[0.01, 0.02, 0.91, 0.02, 0.01, 0.01, 0.01, 0.00, 0.00, 0.01]

This tells you the probability of each digit. argmax finds the INDEX of the highest probability — that index IS the predicted digit (index 2 = digit 2). Without argmax you have 10 probabilities, not a single prediction.

Error 1 — FileNotFoundError: columns.json not found on Render

What happened: Render deployment failed because util.py was looking for files in ./artifacts/ but on Render the files were in Model/
Fix: Changed paths in util.py:

python
# Before (worked locally, failed on Render):
with open('./artifacts/columns.json', 'r') as f:

# After (works on Render):
with open('Model/columns.json', 'r') as f:
Error 2 — Flask serving 404 on Render

What happened: Flask server was running but browser showed 404 — HTML wasn't being served
Fix: Added routes to server.py to serve HTML files:

python
from flask import send_from_directory

@app.route('/')
def serve_index():
    return send_from_directory('../Client', 'app.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('../Client', filename)
Error 3 — Locations dropdown empty on deployed app

What happened: app.js was using localhost URL which doesn't work on Render
Fix: Changed URLs in app.js from absolute to relative:

javascript
// Before (works locally, fails on Render):
var url = "http://127.0.0.1:5000/get_location_names"

// After (works everywhere):
var url = "/get_location_names"
Error 4 — Git: Author identity unknown

What happened: First time using git on laptop — git didn't know who you were
Fix:

bash
git config --global user.email "your@email.com"
git config --global user.name "Gireesh Adireddi"
Error 5 — Git: failed to push some refs

What happened: GitHub repo had content (or wrong branch state) that conflicted with local push
Fix (most reliable method):

bash
git remote remove origin
git remote add origin https://Gireesh08:YOUR_TOKEN@github.com/Gireesh08/repo-name.git
git push -f origin main
# -f = force push — overwrites remote with local
Error 6 — Flask app.run() needs host and port for Render

What happened: app.run() alone only accepts connections from localhost — Render's server couldn't connect
Fix:

python
import os
app.run(host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)))
# host='0.0.0.0' → accept connections from anywhere
# PORT → Render assigns its own port number via environment variable
Correction 1 — Week 5 vs original roadmap order

Misunderstanding: Thought Week 5 was supposed to be CNNs
Correction: Original plan had NLP as Week 5 and CNNs as Week 6. We swapped them. Decision made to continue with CNNs (already in progress) then do NLP in Week 6. All content still covered — different order only.

Correction 2 — File naming case sensitivity caused webpage to show skeleton only

What happened: HTML file was named app.html but other files referenced different casing — CSS and JS didn't load, only raw HTML structure showed
Fix: Ensured all file names matched exactly — app.css, app.js, app.html — no uppercase, matching exactly what the <script src=""> and <link href=""> tags reference