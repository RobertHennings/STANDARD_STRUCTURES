from typing import Dict, List
import os
import config as cfg

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


class StandardProjectStructure(object):
    def __init__(
            self,
            project_path: str=PROJECT_PATH,
            results_path: str=RESULTS_PATH,
            data_path: str=DATA_PATH,
            notebooks_path: str=NOTEBOOKS_PATH,
            models_path: str=MODELS_PATH,
            src_path: str=SRC_PATH,
            util_path: str=UTIL_PATH,
            reports_path: str=REPORTS_PATH,
            reports_figures_path: str=REPORTS_FIGURES_PATH,
            reports_metrics_path: str=REPORTS_METRICS_PATH,
            idea_path: str=IDEA_PATH,
            vs_code_path: str=VS_CODE_PATH,
            literature_path: str=LITERATURE_PATH,
            latex_path: str=LATEX_PATH,
            **kwargs
            ):
            self.project_path=project_path
            self.results_path=results_path
            self.data_path=data_path
            self.notebooks_path=notebooks_path
            self.models_path=models_path
            self.src_path=src_path
            self.util_path=util_path
            self.reports_path=reports_path
            self.reports_figures_path=reports_figures_path
            self.reports_metrics_path=reports_metrics_path
            self.idea_path=idea_path
            self.vs_code_path=vs_code_path
            self.literature_path=literature_path
            self.latex_path=latex_path
            if kwargs:
                for key, value in kwargs.items():
                    setattr(self, key, value)


    def __check_path_existence(
        self,
        path: str
        ):
        """Internal helper method - serves as generous path existence
           checker when saving and reading of an kind of data from files
           suspected at the given location
           
           !!!!If given path does not exist it will be created!!!!

        Args:
            path (str): full path where expected data is saved
        """
        folder_name = path.split("/")[-1]
        path = "/".join(path.split("/")[:-1])
        # FileNotFoundError()
        # os.path.isdir()
        if folder_name not in os.listdir(path):
            print(f"{folder_name} not found in path: {path}")
            folder_path = f"{path}/{folder_name}"
            os.mkdir(folder_path)
            print(f"Folder: {folder_name} created in path: {path}")

    def create_template_project_structure(
            self
            ):
        """Creates a template project structure for a generic software project.
        The structure includes directories for data, notebooks, models,
        source code, utilities, reports, and other relevant folders.
        """
        self.__check_path_existence(path=self.project_path)
        # Get all attributes containing "path"
        all_project_paths = [attribute for attribute in dir(self) if attribute.endswith("path")]
        # Set the general project path to the first position
        # since it contains all the other subfolders
        all_project_paths.remove("project_path")
        all_project_paths.insert(0, "project_path")

        all_project_paths.remove("src_path")
        all_project_paths.insert(1, "src_path")

        all_project_paths.remove("reports_path")
        all_project_paths.insert(2, "reports_path")

        for path_attribute in all_project_paths:
            # Safely retrieve the attribute value
            path = getattr(self, path_attribute, None)
            
            # Check if the path is valid
            if path and isinstance(path, str):
                print(f"path: {path}")
                self.__check_path_existence(path=path)
            else:
                print(f"Invalid or missing path for attribute: {path_attribute}")

    def create_template_university_course_structure(
        self,
        uni_folder_base_path: str=UNI_FOLDER_BASE_PATH,
        semester_list: str=SEMESTER_LIST,
        semester_course_mapping_dict: Dict[str, List[str]]=SEMESTER_COURSE_MAPPING_DICT,
        course_subfolder_list: List[str]=COURSE_SUBFOLDER_LIST,
        create_semester_structure: bool=CREATE_SEMESTER_STRUCTURE,
        create_semester_courses_structure: bool=CREATE_SEMESTER_COURSES_STRUCTURE,
        create_semester_courses_subfolder_structure: bool=CREATE_SEMESTER_COURSES_SUBFOLDER_STRUCTURE
        ):
        """Creates a template university course structure.
        The structure includes directories for each semester and its courses,
        along with optional subfolders for each course.
        """
        self.__check_path_existence(path=uni_folder_base_path)
        if semester_list != []:
            if create_semester_structure:
                for semester in semester_list:
                    semester_path = fr"{uni_folder_base_path}/{semester}"
                    self.__check_path_existence(path=semester_path)
                    if create_semester_courses_structure:
                        if semester_course_mapping_dict[semester] != []: # if course list for semester is not empty, create all subfolders
                            for course in semester_course_mapping_dict[semester]:
                                course_path = fr"{semester_path}/{course}"
                                self.__check_path_existence(path=course_path)
                                if create_semester_courses_subfolder_structure:
                                    for subfolder in course_subfolder_list:
                                        subfolder_path = fr"{course_path}/{subfolder}"
                                        self.__check_path_existence(path=subfolder_path)

# Example Usage 1)
standard_project_structure_instance = StandardProjectStructure()
# Set up the project structure for a generic software project
standard_project_structure_instance.create_template_project_structure()
#  Set up the project structure for a university folder
standard_project_structure_instance.create_template_university_course_structure()


# Example Usage 2)
UNI_FOLDER_BASE_PATH = r"/Users/Robert_Hennings/Uni/Master"
SEMESTER_LIST = [" "]
SEMESTER_COURSE_MAPPING_DICT = {
    "1.Semester": ["Mathematical Finance", "Probability Calculus", "Econometric Methods"],
    "2.Semester": ["Computational Finance", "Inferential Statistics", "Time Series Econometrics", "Reinforcement Learning"],
    "3.Semester": [],
    "4.Semester": [],
    " ": ["Portfolio Analysis"]
}
COURSE_SUBFOLDER_LIST = ["LectureNotes", "ExerciseSheets", "ExamPrep", "HomeAssignments", "Literature", "PC_Tutorial"]
CREATE_SEMESTER_STRUCTURE = True
CREATE_SEMESTER_COURSES_STRUCTURE = True
CREATE_SEMESTER_COURSES_SUBFOLDER_STRUCTURE = True

standard_project_structure_instance.create_template_university_course_structure(
    uni_folder_base_path=UNI_FOLDER_BASE_PATH,
    semester_list=SEMESTER_LIST,
    semester_course_mapping_dict=SEMESTER_COURSE_MAPPING_DICT,
    course_subfolder_list=COURSE_SUBFOLDER_LIST,
    create_semester_structure=CREATE_SEMESTER_STRUCTURE,
    create_semester_courses_structure=CREATE_SEMESTER_COURSES_STRUCTURE,
    create_semester_courses_subfolder_structure=CREATE_SEMESTER_COURSES_SUBFOLDER_STRUCTURE
)