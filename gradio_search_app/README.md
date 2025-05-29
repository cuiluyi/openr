# Gradio Search App

## Overview
The Gradio Search App is a web application that allows users to interactively select and execute various structured search algorithms for problem-solving. Users can input their questions, configure algorithm-specific hyperparameters, and view the results in real-time.

## Features
- Select from multiple structured search algorithms:
  - Cot
  - Best-of-N
  - Vanilla-MCTS
  - MCTS
  - MCTS Beam
- Configure hyperparameters for each algorithm.
- Input questions and receive processed outputs directly in the web interface.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd gradio_search_app
   ```

2. **Install Dependencies**
   Make sure you have Python installed. Then, install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   Start the Gradio application by running:
   ```bash
   python src/app.py
   ```

4. **Access the Web Interface**
   Open your web browser and go to `http://localhost:7860` to access the Gradio interface.

## Usage Example
1. Select an algorithm from the dropdown menu.
2. Enter your question in the input field.
3. Adjust the hyperparameters as needed.
4. Click the "Submit" button to execute the algorithm.
5. View the results displayed on the page.

## Available Algorithms and Hyperparameters

### Cot
- Description: A method that generates answers based on a chain of thought.
- Hyperparameters: 
  - Temperature
  - Max New Tokens

### Best-of-N
- Description: Generates multiple answers and selects the best one.
- Hyperparameters:
  - Number of Sequences
  - Max New Tokens

### Vanilla-MCTS
- Description: A Monte Carlo Tree Search algorithm for decision-making.
- Hyperparameters:
  - Number of Paths
  - Exploration Constant

### MCTS
- Description: An enhanced version of Vanilla-MCTS with additional features.
- Hyperparameters:
  - Number of Paths
  - Exploration Constants

### MCTS Beam
- Description: Combines MCTS with beam search for improved performance.
- Hyperparameters:
  - Beam Size
  - Max Steps

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.