from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import os

# Get the correct path to config files
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
agents_config_path = os.path.join(project_root, "config", "agents.yaml")
tasks_config_path = os.path.join(project_root, "config", "tasks.yaml")

@CrewBase
class SalesContactFinderCrew:
    """Sales Contact Finder crew"""
    
    agents_config = agents_config_path
    tasks_config = tasks_config_path

    @agent
    def company_researcher(self) -> Agent:
        return Agent(
            role="Company Research Specialist",
            goal="Gather comprehensive information about the target company",
            backstory="""You are an expert at researching companies, with a keen eye for details that matter to sales professionals.
            Your task is to gather key information about {target_company} that will be relevant for identifying potential buyers of {our_product}.""",
            allow_delegation=False,
            verbose=True,
        )

    @agent
    def org_structure_analyst(self) -> Agent:
        return Agent(
            role="Organizational Structure Analyst",
            goal="Analyze the company's structure to identify key decision-making roles",
            backstory="""You understand corporate hierarchies and help map decision-makers.
            Analyze {target_company}'s structure and highlight who might influence purchases of {our_product}.""",
            allow_delegation=False,
            verbose=True,
        )

    @agent
    def contact_finder(self) -> Agent:
        return Agent(
            role="Key Contact Identifier",
            goal="Find specific individuals in key roles at the target company",
            backstory="""You are skilled at identifying and locating individuals within organizations.
            Find names, titles, departments, and contact information for decision-makers at {target_company}.""",
            allow_delegation=False,
            verbose=True,
        )

    @agent
    def sales_strategist(self) -> Agent:
        return Agent(
            role="Sales Approach Strategist",
            goal="Develop a strategy for approaching the identified contacts",
            backstory="""You're a sales expert. Craft a tailored outreach strategy for {target_company} that appeals to decision-makers and highlights the value of {our_product}.""",
            allow_delegation=False,
            verbose=True,
        )

    @task
    def research_company_task(self) -> Task:
        return Task(
            description="""Conduct thorough research on {target_company}. Focus on industry, size, recent news, and challenges where {our_product} could help.
            Look for relevant initiatives (digital transformation, procurement revamp, etc.).""",
            expected_output="""## Company Overview
- Brief background on {target_company}
- Key business focus
- Current challenges or initiatives
- Notes on fit for {our_product}""",
            agent=self.company_researcher(),
        )

    @task
    def analyze_org_structure_task(self) -> Task:
        return Task(
            description="""Analyze {target_company}'s organizational structure.
            Identify relevant departments and roles (e.g., CTO, Head of Procurement, Operations).""",
            expected_output="""## Departmental Overview
- List departments relevant to buying decisions
- Explain how each may be involved in purchase of {our_product}""",
            agent=self.org_structure_analyst(),
        )

    @task
    def find_key_contacts_task(self) -> Task:
        return Task(
            description="""Identify individuals at {target_company} with influence over purchasing {our_product}.
            Include title, department, and professional links if possible.""",
            expected_output="""## Key Contacts
| Name | Title | Department | LinkedIn |
|------|-------|------------|----------|
- Aim for 3–5 contacts relevant to the identified roles""",
            agent=self.contact_finder(),
        )

    @task
    def develop_approach_strategy_task(self) -> Task:
        return Task(
            description="""Using all gathered data, develop a contact strategy for {target_company}.
            Focus on messaging and value delivery based on the company's context and contacts' roles.""",
            expected_output="""## Outreach Strategy
- Tailored pitch strategy per persona (e.g., tech leader vs. ops)

## Value Proposition
- Why {our_product} is ideal for {target_company}
- Specific benefits and differentiators

## Summary
Combine all findings into a single clean Markdown document.""",
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
