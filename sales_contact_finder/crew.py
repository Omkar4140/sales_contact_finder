from crewai import Agent, Crew, Process, Task
import os

class SalesContactFinderCrew:
    """Sales Contact Finder crew"""

    def __init__(self):
        self.agents_list = []
        self.tasks_list = []

    def company_researcher(self) -> Agent:
        agent = Agent(
            role="Company Research Specialist",
            goal="Gather comprehensive information about the target company",
            backstory="""You are an expert at researching companies, with a keen eye for details that matter to sales professionals.
            Your task is to gather key information about the target company that will be relevant for identifying potential buyers of the product.""",
            allow_delegation=False,
            verbose=True,
        )
        return agent

    def org_structure_analyst(self) -> Agent:
        agent = Agent(
            role="Organizational Structure Analyst",
            goal="Analyze the company's structure to identify key decision-making roles",
            backstory="""You understand corporate hierarchies and help map decision-makers.
            Analyze the target company's structure and highlight who might influence purchases of the product.""",
            allow_delegation=False,
            verbose=True,
        )
        return agent

    def contact_finder(self) -> Agent:
        agent = Agent(
            role="Key Contact Identifier",
            goal="Find specific individuals in key roles at the target company",
            backstory="""You are skilled at identifying and locating individuals within organizations.
            Find names, titles, departments, and contact information for decision-makers at the target company.""",
            allow_delegation=False,
            verbose=True,
        )
        return agent

    def sales_strategist(self) -> Agent:
        agent = Agent(
            role="Sales Approach Strategist",
            goal="Develop a strategy for approaching the identified contacts",
            backstory="""You're a sales expert. Craft a tailored outreach strategy for the target company that appeals to decision-makers and highlights the value of the product.""",
            allow_delegation=False,
            verbose=True,
        )
        return agent

    def research_company_task(self, company_researcher_agent) -> Task:
        return Task(
            description="""Conduct thorough research on {target_company}. Focus on industry, size, recent news, and challenges where {our_product} could help.
            Look for relevant initiatives (digital transformation, procurement revamp, etc.).""",
            expected_output="""## Company Overview
- Brief background on the target company
- Key business focus
- Current challenges or initiatives
- Notes on fit for the product""",
            agent=company_researcher_agent,
        )

    def analyze_org_structure_task(self, org_analyst_agent) -> Task:
        return Task(
            description="""Analyze {target_company}'s organizational structure.
            Identify relevant departments and roles (e.g., CTO, Head of Procurement, Operations).""",
            expected_output="""## Departmental Overview
- List departments relevant to buying decisions
- Explain how each may be involved in purchase of the product""",
            agent=org_analyst_agent,
        )

    def find_key_contacts_task(self, contact_finder_agent) -> Task:
        return Task(
            description="""Identify individuals at {target_company} with influence over purchasing {our_product}.
            Include title, department, and professional links if possible.""",
            expected_output="""## Key Contacts
| Name | Title | Department | LinkedIn |
|------|-------|------------|----------|
- Aim for 3–5 contacts relevant to the identified roles""",
            agent=contact_finder_agent,
        )

    def develop_approach_strategy_task(self, sales_strategist_agent) -> Task:
        return Task(
            description="""Using all gathered data, develop a contact strategy for {target_company}.
            Focus on messaging and value delivery based on the company's context and contacts' roles.""",
            expected_output="""## Outreach Strategy
- Tailored pitch strategy per persona (e.g., tech leader vs. ops)

## Value Proposition
- Why the product is ideal for the target company
- Specific benefits and differentiators

## Summary
Combine all findings into a single clean Markdown document.""",
            agent=sales_strategist_agent,
            output_file="buyer_contact.md",
        )

    def crew(self) -> Crew:
        """Creates the SalesContactFinder crew"""
        
        # Create agents
        company_researcher_agent = self.company_researcher()
        org_analyst_agent = self.org_structure_analyst()
        contact_finder_agent = self.contact_finder()
        sales_strategist_agent = self.sales_strategist()
        
        # Create tasks
        research_task = self.research_company_task(company_researcher_agent)
        org_task = self.analyze_org_structure_task(org_analyst_agent)
        contact_task = self.find_key_contacts_task(contact_finder_agent)
        strategy_task = self.develop_approach_strategy_task(sales_strategist_agent)
        
        return Crew(
            agents=[company_researcher_agent, org_analyst_agent, contact_finder_agent, sales_strategist_agent],
            tasks=[research_task, org_task, contact_task, strategy_task],
            process=Process.sequential,
            verbose=True,
        )
