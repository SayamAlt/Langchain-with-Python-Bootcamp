from langchain_community.tools import DuckDuckGoSearchRun, ShellTool

# Built-in Tool - DuckDuckGo Search
search_tool = DuckDuckGoSearchRun() # Tools are runnables
results = search_tool.invoke('top news in India today')
print(results)

print(search_tool.name)  # Name of the tool
print(search_tool.description)  # Description of the tool
print(search_tool.args)  # Arguments of the tool
print(search_tool.metadata)  # Metadata of the tool

# Built-in Tool - Shell Tool
shell_tool = ShellTool()
results = shell_tool.invoke('ls -l')
print(results)

# Refer to Langchain documentation for more built-in tools