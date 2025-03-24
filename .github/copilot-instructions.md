### GitHub Copilot Instructions
These instructions are ment for the python part of my master_thesis

### General instructions
- If possible, only return me the corrected code of a very specific part of the whole code, I dont need the whole thing everytime!
- Try to break down big problems into smaller ones and use functions.
- Use a consistent coding style.
- Follow best coding practices.
- Ensure code is efficient and optimized.
- Avoid using deprecated functions or methods.
- Include error handling where appropriate.
- within one paragraph variables are defined, leaving space, like this:
    var_a      = definition_a
    long_var_b = definition_b

### Comments
- Use clear and concise language.
- Always use short comments to explain the purpose of the code. (be scientific)
- comments, explaining a larger part should look like this:
# =============================
# SYSTEM PARAMETERS
# =============================
- comments, explaining one short paragraph should look like this:
### Laser / atom parameters
- comments after code should be done with
code_a = code_b # comment explaining the code

### How to style plots
- Every label of a graphic has to be in latex formatting like : r"$\propto \vec{E}_{\text{out}} / \mu_{\text{a}}$" (variables cursive, descriptions of an object in normal text)
- For Graphics leave out plt.grid(True) and:
- Correctly clip the data to result in a nice figure! and:
- Use one uniform color pallete, that is good for colorblind people: and:
- Write all possible info in the title.
- when displaying floats, please round to an appropriate digit
- always try to display a quantity in terms of another one, to reduce the float values displayed.
- If possible display greek variables with greek letters