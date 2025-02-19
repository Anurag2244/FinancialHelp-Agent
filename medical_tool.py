from crewai_tools import ScrapeWebsiteTool, SerperDevTool

scrape_tool = ScrapeWebsiteTool()

search_tool = SerperDevTool(
    name="search",
    func="Perform a web search",
    description="Fetches search results from the web",
    n_results=10  # Set the number of results to fetch
)