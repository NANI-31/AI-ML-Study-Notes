### 1. Question

A senior data analyst is evaluating a node in a decision tree built for a fraud-detection model. The node contains 40 transactions: 32 are labeled legitimate and 8 are labeled fraudulent.

**What is the Gini impurity of this node, rounded to two decimal places?**

---

### Options

- [ ] **A)** 0.50 — the maximum impurity value, reached only at an exact 50/50 split
- [ ] **B)** 0.20 — the proportion of fraudulent transactions in the node
- [x] **C)** 0.32 — computed from 1 minus the sum of each class proportion squared _(Correct)_
- [ ] **D)** 0.68 — the sum of the squared class proportions before subtracting from 1

---

### Answer Explanation

- **A) Wrong:** 0.50 is the maximum possible impurity for a two-class node, which only occurs at an exact 50/50 split; this node is 80/20, not evenly split.
- **B) Wrong:** 0.20 is simply the fraction of fraudulent transactions (8/40); it skips the squaring-and-subtracting steps of the Gini formula entirely.
- **C) Correct:** $p_{\text{legitimate}} = \frac{32}{40} = 0.8$ and $p_{\text{fraudulent}} = \frac{8}{40} = 0.2$, so:
  $$\text{Gini} = 1 - (0.8^2 + 0.2^2) = 1 - (0.64 + 0.04) = 1 - 0.68 = 0.32$$
- **D) Wrong:** 0.68 is the sum of the squared proportions ($0.64 + 0.04$); it is the intermediate value before subtracting from 1, not the final Gini impurity.

---

### 2. Question

A senior data analyst is walking a new teammate through a diagram of a freshly trained decision tree. The analyst points to the very first node at the top of the diagram — the one that still holds every training sample, before the model has applied a single splitting condition.

**Which term correctly names this node?**

---

### Options

- [x] **A)** Root node — the starting node that holds the entire training dataset before any split _(Correct)_
- [ ] **B)** Decision node — an intermediate node that applies a further splitting condition
- [ ] **C)** Leaf node — the final, terminating node where a class prediction is actually made
- [ ] **D)** Branch node — a labeled connector drawn between two neighboring split conditions

---

### Answer Explanation

- **A) Correct:** The root node is defined as the starting node containing the entire training dataset before any split has been applied, which matches exactly what the analyst is pointing to.
- **B) Wrong:** A decision (intermediate) node applies a further splitting condition to data that has already passed through at least one split; the node described here is before any split.
- **C) Wrong:** A leaf (terminating) node sits at the bottom of the tree and produces the final class prediction, the opposite end from the node described.
- **D) Wrong:** "Branch node" is not a term used for tree structure here; a branch is the connection between nodes, not a node itself.

---

### 3. Question

A backend engineer must build a model to classify customers as "safe" or "risky" for a lending product. A scatter plot of two of the applicant's financial features shows the two classes separated by a curved, non-linear boundary.

**Which modeling choice is best supported by this scatter pattern?**

---

### Options

- [x] **A)** A decision tree classifier, since it builds non-linear boundaries from simple splits _(Correct)_
- [ ] **B)** Logistic regression, since its boundary can bend around any distribution of points
- [ ] **C)** Linear regression, since it predicts the "safe" or "risky" label as a continuous number
- [ ] **D)** A decision tree regressor, since the label is best represented as a continuous score

---

### Answer Explanation

- **A) Correct:** A decision tree builds a non-linear boundary out of a sequence of simple straight splits, which is exactly what a curved, non-linearly separable class boundary requires.
- **B) Wrong:** Logistic regression's prediction is a sigmoid applied to a weighted sum of the inputs, which always produces a straight (linear) decision boundary, not a curved one.
- **C) Wrong:** "Safe" or "risky" is a category, not a continuous quantity, so a regression model that outputs numbers is the wrong tool for this classification task.
- **D) Wrong:** A regressor is built for continuous targets; a categorical safe/risky label calls for a classifier, not a regressor.

---

### 4. Question

A machine learning engineer trains a decision tree classifier on a medical dataset and checks its performance:

```python
print(tree.score(X_train, y_train))
print(tree.score(X_test, y_test))
```

The output is:

```text
1.0
0.87
```

**What does this accuracy gap most strongly indicate about the tree?**

---

### Options

- [ ] **A)** The tree is underfitting, because a test accuracy of 0.87 is too high for a properly trained model
- [ ] **B)** The train_test_split call must be misconfigured, since a gap this large is otherwise impossible
- [ ] **C)** The tree's test accuracy will keep dropping every time the model is retrained on the same data
- [x] **D)** The tree has overfit the training data, memorizing patterns that do not generalize to unseen data _(Correct)_

---

### Answer Explanation

- **A) Wrong:** Underfitting describes a model that performs poorly on both training and test data; a perfect 1.0 training score is the opposite of underfitting.
- **B) Wrong:** A large train/test gap is a normal warning sign of overfitting, not evidence that the data-splitting call itself is broken.
- **C) Wrong:** Nothing about a single train/test evaluation predicts that accuracy will keep falling on future retraining; this is an unsupported claim.
- **D) Correct:** A training accuracy of 1.0 alongside a noticeably lower test accuracy is the classic signature of overfitting: the tree has memorized training patterns that do not carry over to unseen data.

---

### 5. Question

A product manager is training new support staff and wants a familiar comparison for how a decision tree model arrives at a prediction: a sequence of simple, binary questions where each answer routes the data point down a different path.

**Which scenario best mirrors how a decision tree makes its prediction?**

---

### Options

- [ ] **A)** A GPS app that continuously compares distances across many possible driving routes
- [x] **B)** An IVR phone system that routes each call down a different branch based on which button is pressed _(Correct)_
- [ ] **C)** A stock ticker that continuously updates a company's share price throughout the trading day
- [ ] **D)** A search engine that ranks web pages using a continuously computed relevance score

---

### Answer Explanation

- **A) Wrong:** Comparing many continuous distances is not a sequence of discrete binary questions; it does not route the input down one of two branches at each step.
- **B) Correct:** Pressing 1 for one department or 2 for another routes the caller down a different branch, one binary choice at a time, which is exactly how each decision-tree split routes a data point.
- **C) Wrong:** A continuously updating price is a single evolving number, not a sequence of binary routing decisions.
- **D) Wrong:** A continuously computed relevance score ranks items along one scale; it does not branch a data point through a series of yes/no questions.

---

### 6. Question

A senior data analyst is deciding which feature a decision tree should split on at a node in a new customer-churn dataset. Three candidate features were each tested at their own best threshold:

| Candidate feature     | Best threshold | Weighted Gini |
| :-------------------- | :------------- | :------------ |
| tenure_months         | <= 8           | 0.39          |
| monthly_charges       | <= 65          | 0.44          |
| contract_type_encoded | <= 1           | 0.41          |

**Which statements are correct about choosing the split variable?**

---

### Options

- [x] **A)** `tenure_months` should become the split at this node, because it produces the lowest weighted Gini among the three candidates _(Correct)_
- [x] **B)** The algorithm compares every candidate variable and threshold, then selects the combination with the lowest weighted Gini as the split _(Correct)_
- [ ] **C)** `monthly_charges` should become the split, because its higher weighted Gini shows it produces the greatest information gain
- [ ] **D)** Because three different features were tested here, this node must use a fundamentally different splitting algorithm than the one used earlier

---

### Answer Explanation

- **A) Correct:** 0.39 is the lowest of the three weighted Gini values, and the variable whose best split produces the lowest weighted Gini is the one selected for that node.
- **B) Correct:** At every node, every candidate variable and every candidate threshold for that variable are compared, and the combination with the lowest weighted Gini (highest information gain) is chosen as the split condition.
- **C) Wrong:** A higher weighted Gini means a smaller drop from the parent's Gini, so it represents the least information gain among the candidates, not the greatest.
- **D) Wrong:** Evaluating multiple candidate variables and thresholds at a node is the standard split-search procedure used at every node of a decision tree; it does not indicate a different algorithm.

---

### 7. Question

A senior data analyst is reviewing a diagram of a freshly trained decision tree with a colleague.

**Which statements correctly describe the tree's structure?**

---

### Options

- [ ] **A)** Leaf nodes are also called decision nodes, since both make the final class prediction
- [ ] **B)** The diagram is conventionally drawn with the root at the bottom and the leaves at the top
- [x] **C)** The root node contains the entire training dataset before any split has been applied _(Correct)_
- [x] **D)** Decision nodes, also called intermediate nodes, apply a further splitting condition to incoming data _(Correct)_

---

### Answer Explanation

- **A) Wrong:** Leaf nodes are also called terminating nodes and are where a prediction is made, but decision (intermediate) nodes apply further splits rather than final predictions; the two terms are not interchangeable.
- **B) Wrong:** A trained decision tree is conventionally drawn as an upside-down tree, with the root at the top and the leaf nodes at the bottom, the reverse of this statement.
- **C) Correct:** The root node is explicitly defined as the starting node holding the entire training dataset before any split is applied.
- **D) Correct:** Decision nodes (also called intermediate nodes) are the nodes that apply a further splitting condition to the data that reaches them.

---

### 8. Question

A senior data analyst is evaluating a candidate split for a churn-prediction decision tree. The root node contains 500 customers: 350 who did not churn and 150 who churned, giving $\text{Gini}(\text{root}) = 1 - (0.7^2 + 0.3^2) = 0.42$.

A candidate split on tenure produces:

- **Left child:** $N = 200$, not-churned = 170, churned = 30  
  $p_{\text{not-churned}} = 0.85$, $p_{\text{churned}} = 0.15$  
  $\text{Gini}(\text{left}) = 1 - (0.85^2 + 0.15^2) = 0.255$

- **Right child:** $N = 300$, not-churned = 180, churned = 120  
  $p_{\text{not-churned}} = 0.6$, $p_{\text{churned}} = 0.4$  
  $\text{Gini}(\text{right}) = 1 - (0.6^2 + 0.4^2) = 0.48$

**Which statements are correct about this split?**

---

### Options

- [ ] **A)** The left child is more impure than the right child, since it holds fewer of the churned customers
- [ ] **B)** This split should be rejected because it increases impurity compared to the parent node
- [x] **C)** The weighted Gini of this split, combining both child nodes by sample size, comes out to approximately 0.39 _(Correct)_
- [x] **D)** The information gain of this split, the drop from the parent's Gini to the weighted child Gini, is approximately 0.03 _(Correct)_

---

### Answer Explanation

- **A) Wrong:** Impurity is measured by the Gini value, not by how many churned customers a branch holds; the left child's Gini of 0.255 is lower than the right child's 0.48, so the left child is purer, not more impure.
- **B) Wrong:** The weighted Gini of the split (0.39) is lower than the parent's Gini (0.42), which means the split reduces impurity and produces positive information gain, so it should not be rejected.
- **C) Correct:** Weighted Gini = $\frac{200 \times 0.255 + 300 \times 0.48}{500} = \frac{51 + 144}{500} = \frac{195}{500} = 0.39$.
- **D) Correct:** Information gain = $\text{Gini}(\text{parent}) - \text{Weighted Gini}(\text{children}) = 0.42 - 0.39 = 0.03$.

---

### 9. Question

A data scientist trains a decision tree on all 30 available measurements for a tumor classification task and observes 100% training accuracy but only about 90% test accuracy. To reduce this gap, the data scientist retrains the same classifier using only the first 10 of the 30 measurements and compares the new test accuracy against the original.

**Which outcome would best support the idea that fewer features reduce this overfitting gap?**

---

### Options

- [ ] **A)** The retrained tree's training accuracy drops to 50 percent, showing it can no longer learn from the data
- [x] **B)** The retrained tree's test accuracy rises closer to its training accuracy, narrowing the original gap _(Correct)_
- [ ] **C)** The retrained tree's test accuracy falls to 0 percent, showing that removing features always breaks a tree
- [ ] **D)** The retrained tree produces the exact same training and test accuracy as the original model

---

### Answer Explanation

- **A) Wrong:** A collapse in training accuracy would suggest the tree can no longer fit the data at all, which indicates underfitting rather than a reduction in overfitting.
- **B) Correct:** Using all 30 features allowed the tree to grow deep enough to memorize the training set; a narrower gap between test and training accuracy after reducing features directly supports the idea that fewer features reduce overfitting.
- **C) Wrong:** A complete drop to 0 percent test accuracy is an extreme, implausible failure indicating a broken model, not a successful fix for an overfitting gap.
- **D) Wrong:** Identical accuracy values across both versions would show that reducing the feature count had no impact, contradicting the hypothesis rather than supporting it.

---

### 10. Question

A product manager and a data scientist are debating why decision trees use Gini impurity instead of the mean squared error used in linear and logistic regression, and want to settle which of the following claims about that choice are correct.

**Which statements about Gini impurity are correct?**

---

### Options

- [x] **A)** Gini impurity replaces mean squared error, since MSE does not suit classification _(Correct)_
- [x] **B)** For any node with exactly two classes, Gini impurity can never exceed 0.5 _(Correct)_
- [ ] **C)** A Gini impurity value of 0.5 means the node is perfectly pure
- [ ] **D)** The Gini impurity formula only works when a node contains exactly two classes

---

### Answer Explanation

- **A) Correct:** The mean squared error used for linear and logistic regression does not apply to this classification setting, so Gini impurity is introduced as the metric for scoring split quality instead.
- **B) Correct:** For a two-class node, Gini impurity ranges from 0 (perfectly pure) up to 0.5, which occurs only at an exact 50/50 split.
- **C) Wrong:** A value of 0 means perfectly pure and 0.5 means maximally impure; this statement reverses the two boundary meanings.
- **D) Wrong:** The general Gini impurity formula equals 1 minus the sum of the squared proportions of however many classes are present, so it works for any number of classes, not only two.
