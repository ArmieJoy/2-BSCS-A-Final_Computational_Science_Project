import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import _tree
import plotly.graph_objects as go #type:ignore
import numpy as np

# ---- Load dataset ----
df = pd.read_excel(r"C:\Users\ASUS\OneDrive\Documents\GitHub\2-BSCS-A-Final_Computational_Science_Project\disaster\decisiontree.xlsx")
df.columns = df.columns.str.strip()

print("DATA PREVIEW:")
print(df.head())

# ---- Encode categorical columns ----
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ---- Define target ----
target_column = "Disaster Severity"
X = df.drop(target_column, axis=1)
y = df[target_column]

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Train Decision Tree ----
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

print("Model trained successfully!")

# ---- Build Plotly Tree ----
def build_plotly_tree(model, feature_names):
    import numpy as np
    import plotly.graph_objects as go #type:ignore
    from sklearn.tree import _tree

    tree_ = model.tree_
    TREE_UNDEFINED = _tree.TREE_UNDEFINED
    n_nodes = tree_.node_count

    x_pos, y_pos, labels, hover_text = {}, {}, {}, {}
    is_leaf = np.zeros(n_nodes, dtype=bool)
    x_counter = [0]

    # --- Walk tree to assign positions ---
    def walk(node, depth=0):
        y_pos[node] = -depth
        if tree_.feature[node] == TREE_UNDEFINED:
            is_leaf[node] = True
            x_pos[node] = x_counter[0]
            x_counter[0] += 1
            value = tree_.value[node][0]
            predicted_class = int(value.argmax())
            labels[node] = f"Leaf<br>Class {predicted_class}"
            hover_text[node] = (
                f"Leaf Node<br>"
                f"Samples: {tree_.n_node_samples[node]}<br>"
                f"Class Counts: {value}<br>"
                f"Predicted Class: {predicted_class}"
            )
            return
        feat = feature_names[tree_.feature[node]]
        thresh = tree_.threshold[node]
        # Wrap text so it fits inside rectangle
        labels[node] = f"{feat}<br>≤ {thresh:.2f}"
        hover_text[node] = (
            f"Node {node}<br>"
            f"Feature: {feat}<br>"
            f"Threshold: ≤ {thresh:.2f}<br>"
            f"Samples: {tree_.n_node_samples[node]}"
        )
        left, right = tree_.children_left[node], tree_.children_right[node]
        if left != TREE_UNDEFINED:
            walk(left, depth + 1)
        if right != TREE_UNDEFINED:
            walk(right, depth + 1)
        if left != TREE_UNDEFINED and right != TREE_UNDEFINED:
            x_pos[node] = (x_pos[left] + x_pos[right]) / 2
        else:
            x_pos[node] = x_pos[left] if left != TREE_UNDEFINED else x_pos[right]

    walk(0)

    # --- Layout scaling ---
    x_scale, y_scale = 4.0, 2.5   # more spacing so nodes don’t overlap

    fig = go.Figure()

    # --- Edges (stems) ---
    edge_x, edge_y = [], []
    for node in range(n_nodes):
        for child in (tree_.children_left[node], tree_.children_right[node]):
            if child != TREE_UNDEFINED and child in x_pos:
                edge_x += [x_pos[node] * x_scale, x_pos[child] * x_scale, None]
                edge_y += [y_pos[node] * y_scale, y_pos[child] * y_scale, None]
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="gray", width=2),
        hoverinfo="skip",
        showlegend=False
    ))

    # --- Rectangles + text ---
    for node in range(n_nodes):
        cx, cy = x_pos[node] * x_scale, y_pos[node] * y_scale

        # Smaller boxes for deeper nodes (like 10, 18, 21, 26)
        depth = -y_pos[node]
        if is_leaf[node]:
            box_w, box_h = 1.8, 0.6
            fill = "lightgreen"
        else:
            # shrink boxes gradually with depth
            box_w = max(2.5, 3.5 - 0.3*depth)
            box_h = max(0.6, 0.8 - 0.05*depth)
            fill = "lightblue"

        fig.add_shape(
            type="rect",
            x0=cx - box_w/2, x1=cx + box_w/2,
            y0=cy - box_h/2, y1=cy + box_h/2,
            line=dict(color="black", width=1),
            fillcolor=fill,
            layer="above"
        )
        fig.add_annotation(
            x=cx, y=cy,
            text=labels[node],
            showarrow=False,
            font=dict(size=11, color="black"),
            align="center",
            xanchor="center", yanchor="middle"
        )

    # --- Invisible markers for hover info ---
    fig.add_trace(go.Scatter(
        x=[x_pos[n]*x_scale for n in range(n_nodes)],
        y=[y_pos[n]*y_scale for n in range(n_nodes)],
        mode="markers",
        marker=dict(size=2, color="rgba(0,0,0,0)"),
        hoverinfo="text",
        text=[hover_text[n] for n in range(n_nodes)],
        showlegend=False
    ))

    fig.update_layout(
        title="🌳 Compact Balanced Decision Tree",
        dragmode="zoom",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        width=1800, height=1000,
        font=dict(size=11)
    )
    fig.show()


# ---- Run visualization ----
build_plotly_tree(model, X.columns)
