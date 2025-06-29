# Research Autominer 🔬📊

  ```
  ```
  /\/\/\/\/\/\/\/\/
   / /\/\/\/\/\/\/\/\/ \
  | |            | |
  | |            | |
  | |            | |
  | |            | |
  | |            | |
   \ \/\/\/\/\/\/\/ /\
    \/\/\/\/\/\/\/\/
```
```



## Empowering Medical Research with Automated Data Processing

Research Autominer is a cutting-edge application designed to revolutionize medical research by automating complex data processing workflows. Built with a focus on user-friendliness and efficiency, it transforms raw medical data into actionable insights, enabling doctors and researchers to accelerate discoveries and advance the field of medicine.

### Background

Traditional medical research often involves tedious and time-consuming manual data handling, prone to human error and difficult to reproduce. Research Autominer addresses this challenge by providing a robust, intuitive platform for data mining. It originated from the need to streamline the analysis of large datasets, allowing medical professionals to focus more on interpretation and less on preparation. This software has already been instrumental in several publications, demonstrating its real-world impact and reliability in clinical research settings.

### The Role of AI and Automation

Research Autominer leverages the power of Artificial Intelligence, specifically Gemini AI, to enhance the research workflow. It's crucial to clarify that **AI in no stage touches the raw data or performs any data analysis directly.** This strict separation ensures data safety, integrity, and compliance with medical research standards.

Instead, AI is strategically utilized for:

*   **Automated Code Generation:** AI assists in creating well-defined and manually vetted Python code snippets that automate data processing tasks as specified by the researcher. This significantly reduces the time and effort required to write complex scripts.
*   **Explanation of Analysis Steps:** AI can explain various data analysis steps, making complex methodologies more accessible and understandable for researchers, especially those without extensive programming backgrounds.

The core benefit of Research Autominer lies in its automation capabilities:

*   **Minimizing Human Error:** By automating repetitive tasks, the software drastically reduces the potential for errors associated with manual data manipulation in spreadsheets.
*   **Reproducibility and Re-runability:** Workflows are saved as "manuscripts," which are fully reproducible. This means that if new or updated datasets become available, researchers can re-run the entire analysis with minimal effort, avoiding redundant work and incentivizing continuous data extraction and refinement. This ensures that research findings can be easily validated and updated as new information emerges.

### Features

*   **Intuitive Graphical User Interface (GUI):** Transition from script-based workflows to a user-friendly web interface.
*   **Flexible Data Input:** Easily load CSV files, preview column headers, and navigate through paginated patient data.
*   **"Manuscript" Management:** Create, save, and load reproducible data processing workflows ("manuscripts") with clear, descriptive steps.
*   **Live Testing:** Test manuscript steps on sample data with immediate visual feedback.
*   **AI-Assisted Function Generation:** Utilize Gemini AI to help generate custom data processing functions.
*   **Dynamic Column Control:** Show/hide columns, and truncate content for better data visualization.

### Usage

To get started with Research Autominer, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ArielRotem/ResearchAutominer2.0.git
    cd ResearchAutominer2.0
    ```

2.  **Set up your Gemini API Key:**
    Open `backend/api/gemini_client.py` and replace `"YOUR_API_KEY"` with your actual Gemini API key.

3.  **Run the application:**
    This script will start both the FastAPI backend and the React development server.
    ```bash
    ./run_app.sh
    ```
    The application will typically be accessible at `http://localhost:3000` in your web browser.

4.  **Explore the features:**
    *   Load `data/input.csv` in the Data Preview section.
    *   Experiment with the paginated data view and column visibility controls.
    *   Load `manuscripts/ResearchCT_Full_Workflow.json` in the Manuscript Editor to see a comprehensive example workflow.
    *   Interact with the dynamic parameter inputs (lists, dictionaries, column selectors).
    *   Test manuscript steps on sample data using the Live Data Test window.

### Author

**Ariel Rotem**

This project is driven by the vision to significantly advance medical research and the broader scientific field through innovative software solutions. Research Autominer is actively used by real doctors and has contributed to several publications, underscoring its practical utility and impact.

### Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.


