def evaluate_model():
    plot_sample_predictions_decomposed(model_one4all,
                                        seq_len, test_loader_actual,
                                            num_samples=3, device='cpu')
    plt.show()