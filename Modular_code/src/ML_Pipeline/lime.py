import lime
import lime.lime_tabular

# define a function for lime - local interpretable model-agnostic explanations
def lime_explanation(model, X_train, X_test, class_names, chosen_index):
    # Initialize the LimeTabularExplainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        class_names=class_names,
        kernel_width=5
    )
    
    # Select the instance from the test set
    chosen_instance = X_test.iloc[chosen_index].values  # Correctly select by index
    
    # Generate explanation for the chosen instance
    exp = explainer.explain_instance(
        chosen_instance,
        lambda x: model.predict_proba(x).astype(float),
        num_features=10
    )
    
    # Generate and return the explanation figure
    fig = exp.as_pyplot_figure()
    return fig
