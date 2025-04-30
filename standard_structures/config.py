# standard_project_structure
PROJECT_PATH = r"/Users/Robert_Hennings/Projects/TEST_PROJECT"
RESULTS_PATH = rf"{PROJECT_PATH}/results"
HTML_GRAPHS_PATH = rf"{RESULTS_PATH}/html_graphs" # store produced html graphs
DATA_PATH  = rf"{PROJECT_PATH}/data"
NOTEBOOKS_PATH = rf"{PROJECT_PATH}/notebooks" # store jupyter notebooks for exploration
MODELS_PATH = rf"{PROJECT_PATH}/models"
SRC_PATH = rf"{PROJECT_PATH}/src"
UTIL_PATH = rf"{SRC_PATH}/util"
REPORTS_PATH = rf"{PROJECT_PATH}/reports" # store model logs and reports from training/testing
REPORTS_FIGURES_PATH = rf"{REPORTS_PATH}/html_graphs"
REPORTS_METRICS_PATH = rf"{REPORTS_PATH}/metrics"
IDEA_PATH = rf"{PROJECT_PATH}/idea" # store the 
VS_CODE_PATH = rf"{PROJECT_PATH}/.vscode" # store the vs code extensions.json and settings.json
LITERATURE_PATH = rf"{PROJECT_PATH}/literature"
LATEX_PATH = rf"{PROJECT_PATH}/latex_project" # store the latex related files for the article
# University structure
UNI_FOLDER_BASE_PATH = r"/Users/Robert_Hennings/Uni/test"
SEMESTER_LIST = ["1.Semester", "2.Semester", "3.Semester", "4.Semester"]
SEMESTER_COURSE_MAPPING_DICT = {
    "1.Semester": ["Mathematical Finance", "Probability Calculus", "Econometric Methods"],
    "2.Semester": ["Computational Finance", "Inferential Statistics", "Time Series Econometrics", "Reinforcement Learning"],
    "3.Semester": [],
    "4.Semester": []
}
COURSE_SUBFOLDER_LIST = ["LectureNotes", "ExerciseSheets", "ExamPrep", "HomeAssignments", "Literature", "PC_Tutorial"]
CREATE_SEMESTER_STRUCTURE = True
CREATE_SEMESTER_COURSES_STRUCTURE = True
CREATE_SEMESTER_COURSES_SUBFOLDER_STRUCTURE = True

# standard_data_loading

# standard_data_processing

# standard_data_plotting