#from crewai_tools import SerperDevTool

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase(
    agents_config_path="config/agents.yaml",
    tasks_config_path="config/tasks.yaml"
)

@CrewBase
class SalesContactFinderCrew:

    @agent
    def company_researcher(self) -> Agent:
        return Agent(
            config="company_researcher",
            tools=[],
            allow_delegation=False,
            verbose=True,
        )

    @agent
    def org_structure_analyst(self) -> Agent:
        return Agent(
            config="org_structure_analyst",
            tools=[],
            allow_delegation=False,
            verbose=True,
        )

    @agent
    def contact_finder(self) -> Agent:
        return Agent(
            config="contact_finder",
            tools=[],
            allow_delegation=False,
            verbose=True,
        )

    @agent
    def sales_strategist(self) -> Agent:
        return Agent(
            config="sales_strategist",
            tools=[],
            allow_delegation=False,
            verbose=True,
        )

    @task
    def research_company_task(self) -> Task:
        return Task(
            config="research_company_task",
            agent=self.company_researcher(),
        )

    @task
    def analyze_org_structure_task(self) -> Task:
        return Task(
            config="analyze_org_structure_task",
            agent=self.org_structure_analyst(),
        )

    @task
    def find_key_contacts_task(self) -> Task:
        return Task(
            config="find_key_contacts_task",
            agent=self.contact_finder(),
        )

    @task
    def develop_approach_strategy_task(self) -> Task:
        return Task(
            config="develop_approach_strategy_task",
            agent=self.sales_strategist(),
            output_file="buyer_contact.md",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the SalesContactFinder crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            
        )
