from file_handling import LuciFileHandler
from class_extraction import LuciClassExtractor
from description_generation import LuciDescriptionGenerator
from jsonl_conversion import LuciJSONLConverter
from langchain_community.llms import Ollama
from langchain_openai import OpenAI


def main(source_directory, source_filename, output_directory):
    # Initialize FileHandler and read the source file
    file_handler = LuciFileHandler(source_directory, output_directory)
    content = file_handler.read_file(source_filename)

    # Extract classes and write each to a separate file
    extractor = LuciClassExtractor(content)
    classes = extractor.extract_classes()
    for class_header, class_body in classes:  # Fixed iteration here
        class_name = class_header.split()[1]  # Extracting class name from the header
        file_handler.write_file(f"{class_header}\n    {class_body}", f"{class_name}.py")

    # Initialize the description generator with the Ollama model
    ollama_model = Ollama(model='llama2:70b')
    description_generator = LuciDescriptionGenerator(output_directory, ollama_model, output_directory)
    description_generator.generate_descriptions()

    # Convert descriptions and classes into a JSONL file
    jsonl_converter = LuciJSONLConverter(output_directory, output_directory)
    jsonl_converter.convert_to_jsonl()

if __name__ == "__main__":
    main("manim_files", "gpt3.py.source", "output")

