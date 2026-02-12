## Repository for classes of the course AI506 Advanced Machine Learning @ SDU

### Course links

- [Course page](https://odin.sdu.dk/sitecore/index.php?a=fagbesk&id=154261&listid=37232&lang=en)

### Structure of the repository

The structure of the repository is as follows:
- For each year there is a folder
- Each year folder contains folders for each class in that year (e.g. class_01, class_02, etc.)
- Each class folder contains the materials for that class usually divided into docs (e.g. slides, exercises texts) and code folders (e.g. solutions, example code)

### Data handling

Since different code could use same data, we have a common folder for data. This is done to prevent from downloading the same data multiple times and filling up storage.

It is assumed the data is stored in the `shared_data` folder (create it if not present), in the project root. You can access the path by importing it from the `constants.py` file, in the project root.

### Teachers and TAs
- Teachers: [Lukas Galke Poech](https://portal.findresearcher.sdu.dk/en/persons/lukas-paul-achatius-galke-poech/)
- Teaching assistants: [Gianluca Barmina](https://portal.findresearcher.sdu.dk/en/persons/gbarmina/), [Mogens From](https://portal.findresearcher.sdu.dk/en/persons/mogens-from/)