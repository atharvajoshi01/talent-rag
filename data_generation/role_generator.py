"""
Role Data Generator Module.

This module generates synthetic job role data including descriptions,
requirements, and qualifications for testing the Talent RAG system.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field, asdict

from faker import Faker
from loguru import logger


# Role templates organized by category
ROLE_TEMPLATES = {
    "ml_engineer": {
        "titles": [
            "Machine Learning Engineer",
            "ML Platform Engineer",
            "Applied ML Engineer",
            "ML Infrastructure Engineer",
            "Deep Learning Engineer"
        ],
        "required_skills": [
            ["Python", "TensorFlow", "PyTorch"],
            ["Python", "MLOps", "Kubernetes"],
            ["Python", "Deep Learning", "Computer Vision"],
            ["Python", "NLP", "Hugging Face"],
            ["Python", "LLMs", "RAG Systems"]
        ],
        "nice_to_have": [
            "Distributed training", "Model optimization", "CUDA programming",
            "Ray", "Spark MLlib", "Feature stores", "A/B testing"
        ]
    },
    "data_scientist": {
        "titles": [
            "Data Scientist",
            "Senior Data Scientist",
            "Staff Data Scientist",
            "Applied Scientist",
            "Research Scientist"
        ],
        "required_skills": [
            ["Python", "SQL", "Statistics"],
            ["Python", "Machine Learning", "A/B Testing"],
            ["Python", "Causal Inference", "Experimentation"],
            ["Python", "R", "Data Analysis"],
            ["Python", "Deep Learning", "Research"]
        ],
        "nice_to_have": [
            "PhD in quantitative field", "Published research", "Spark",
            "Time series analysis", "Bayesian methods", "Reinforcement learning"
        ]
    },
    "backend_engineer": {
        "titles": [
            "Backend Engineer",
            "Software Engineer - Backend",
            "Platform Engineer",
            "Infrastructure Engineer",
            "Distributed Systems Engineer"
        ],
        "required_skills": [
            ["Python", "PostgreSQL", "Redis"],
            ["Go", "gRPC", "Kubernetes"],
            ["Java", "Spring Boot", "Kafka"],
            ["Python", "Django", "AWS"],
            ["Node.js", "MongoDB", "Docker"]
        ],
        "nice_to_have": [
            "Microservices", "Event-driven architecture", "GraphQL",
            "Message queues", "Load balancing", "Caching strategies"
        ]
    },
    "fullstack_engineer": {
        "titles": [
            "Full Stack Engineer",
            "Software Engineer",
            "Product Engineer",
            "Frontend Engineer",
            "Web Developer"
        ],
        "required_skills": [
            ["React", "Node.js", "TypeScript"],
            ["Vue.js", "Python", "PostgreSQL"],
            ["Angular", "Java", "MongoDB"],
            ["Next.js", "GraphQL", "AWS"],
            ["React", "Django", "PostgreSQL"]
        ],
        "nice_to_have": [
            "Mobile development", "Performance optimization", "Accessibility",
            "Design systems", "Testing frameworks", "CI/CD"
        ]
    },
    "engineering_manager": {
        "titles": [
            "Engineering Manager",
            "Technical Lead Manager",
            "Director of Engineering",
            "VP of Engineering",
            "Head of Engineering"
        ],
        "required_skills": [
            ["People Management", "Technical Architecture", "Agile"],
            ["Team Leadership", "System Design", "Stakeholder Management"],
            ["Engineering Strategy", "Budget Management", "Hiring"],
            ["Technical Vision", "Cross-functional Leadership", "OKRs"],
            ["Organization Design", "Process Improvement", "Mentoring"]
        ],
        "nice_to_have": [
            "Prior IC experience", "Startup experience", "MBA",
            "Executive coaching", "Public speaking", "Open source contributions"
        ]
    }
}

COMPANIES = [
    "TechCorp", "DataDriven Inc", "AI Solutions", "CloudFirst",
    "InnovateTech", "ScaleUp Systems", "NextGen AI", "DataFlow",
    "Quantum Labs", "Neural Networks Inc", "Apex Engineering",
    "Velocity Tech", "Horizon AI", "Summit Solutions", "EdgeTech"
]

LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
    "Boston, MA", "Los Angeles, CA", "Denver, CO", "Chicago, IL",
    "Remote (US)", "Hybrid - San Francisco", "Hybrid - New York"
]

SENIORITY_LEVELS = ["Junior", "Mid-Level", "Senior", "Staff", "Principal", "Director"]

TEAM_CONTEXTS = [
    "fast-growing startup disrupting the {industry} industry",
    "established tech company with a strong engineering culture",
    "innovative AI research lab pushing the boundaries of machine learning",
    "high-performance team building next-generation {product} solutions",
    "mission-driven organization focused on {mission}",
    "rapidly scaling platform serving millions of users globally"
]

INDUSTRIES = [
    "fintech", "healthcare", "e-commerce", "education", "entertainment",
    "logistics", "social media", "enterprise software", "autonomous vehicles"
]

PRODUCTS = [
    "data infrastructure", "ML platform", "analytics", "recommendation",
    "search", "automation", "developer tools", "security"
]

MISSIONS = [
    "democratizing AI", "improving healthcare outcomes",
    "sustainable technology", "financial inclusion",
    "educational access", "workplace productivity"
]


@dataclass
class Role:
    """
    Complete job role definition.

    Attributes:
        id: Unique role identifier
        title: Job title
        company: Company name
        location: Job location
        seniority: Required seniority level
        required_skills: List of required technical skills
        nice_to_have_skills: List of preferred skills
        years_experience_required: Minimum years of experience
        description: Full job description
        responsibilities: List of key responsibilities
        qualifications: List of required qualifications
        benefits: List of benefits offered
        salary_range: Salary range as string
        created_at: Timestamp of role creation
    """
    id: str
    title: str
    company: str
    location: str
    seniority: str
    required_skills: list[str]
    nice_to_have_skills: list[str]
    years_experience_required: int
    description: str
    responsibilities: list[str]
    qualifications: list[str]
    benefits: list[str]
    salary_range: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert role to dictionary representation."""
        return asdict(self)


class RoleGenerator:
    """
    Generator for synthetic job role data.

    This class generates realistic job postings with descriptions,
    requirements, and qualifications for testing recruitment AI systems.

    Attributes:
        faker: Faker instance for generating random data
        seed: Random seed for reproducibility
    """

    def __init__(self, seed: int | None = None):
        """
        Initialize the role generator.

        Args:
            seed: Optional random seed for reproducibility
        """
        self.faker = Faker()
        if seed is not None:
            # seed_instance scopes the seed to this Faker instance so two
            # generators constructed with the same seed produce identical
            # output. Faker.seed() seeds the shared Generator, which makes
            # later constructions interfere with earlier ones.
            self.faker.seed_instance(seed)
            self._random = random.Random(seed)
        else:
            self._random = random
        self.seed = seed
        logger.info(f"RoleGenerator initialized with seed={seed}")

    def _get_salary_range(self, seniority: str, role_category: str) -> str:
        """
        Generate realistic salary range based on seniority and role.

        Args:
            seniority: Role seniority level
            role_category: Category of the role

        Returns:
            Salary range string
        """
        base_ranges = {
            "Junior": (80000, 120000),
            "Mid-Level": (120000, 180000),
            "Senior": (160000, 240000),
            "Staff": (200000, 300000),
            "Principal": (250000, 380000),
            "Director": (300000, 450000)
        }

        # ML roles tend to pay higher
        multiplier = 1.0
        if role_category in ["ml_engineer", "data_scientist"]:
            multiplier = 1.15
        elif role_category == "engineering_manager":
            multiplier = 1.1

        min_salary, max_salary = base_ranges.get(seniority, (100000, 150000))
        min_salary = int(min_salary * multiplier)
        max_salary = int(max_salary * multiplier)

        return f"${min_salary // 1000}K - ${max_salary // 1000}K + equity"

    def _generate_responsibilities(
        self,
        role_category: str,
        seniority: str,
        skills: list[str]
    ) -> list[str]:
        """
        Generate role-specific responsibilities.

        Args:
            role_category: Category of the role
            seniority: Seniority level
            skills: Required skills for context

        Returns:
            List of responsibility strings
        """
        base_responsibilities = {
            "ml_engineer": [
                "Design and implement production ML systems at scale",
                f"Build and optimize {self._random.choice(['deep learning', 'NLP', 'computer vision'])} models",
                "Develop ML pipelines for training, validation, and deployment",
                "Collaborate with product teams to identify ML opportunities",
                "Monitor model performance and implement improvements",
                "Create reusable ML components and best practices documentation",
                "Conduct experiments and A/B tests to validate model improvements"
            ],
            "data_scientist": [
                "Analyze large datasets to derive actionable business insights",
                "Build predictive models to drive product decisions",
                "Design and analyze A/B experiments",
                "Collaborate with stakeholders to define success metrics",
                "Create dashboards and reports for executive visibility",
                "Develop statistical models for causal inference",
                "Present findings to technical and non-technical audiences"
            ],
            "backend_engineer": [
                "Design and build scalable backend services and APIs",
                "Optimize system performance and reliability",
                "Implement data storage solutions and caching strategies",
                "Write clean, testable, and well-documented code",
                "Participate in code reviews and architectural discussions",
                "Debug and resolve production issues",
                "Mentor junior engineers and contribute to team growth"
            ],
            "fullstack_engineer": [
                "Build responsive and performant user interfaces",
                "Develop backend APIs and database schemas",
                "Collaborate with designers on user experience",
                "Implement end-to-end features from concept to deployment",
                "Write automated tests for frontend and backend",
                "Optimize application performance and accessibility",
                "Contribute to technical documentation and standards"
            ],
            "engineering_manager": [
                "Lead and grow a team of high-performing engineers",
                "Define technical strategy and roadmap",
                "Drive hiring and team composition decisions",
                "Foster a culture of engineering excellence",
                "Partner with product and design on prioritization",
                "Remove blockers and enable team productivity",
                "Conduct performance reviews and career development"
            ]
        }

        # Add seniority-specific responsibilities
        seniority_additions = {
            "Senior": [
                "Mentor junior and mid-level team members",
                "Lead technical design discussions and code reviews",
                "Drive adoption of best practices across the team"
            ],
            "Staff": [
                "Define technical direction for multi-team initiatives",
                "Represent engineering in cross-functional leadership forums",
                "Lead architecture reviews for critical systems"
            ],
            "Principal": [
                "Set technical vision across the organization",
                "Drive industry-leading innovations",
                "Influence company-wide technical strategy"
            ],
            "Director": [
                "Manage multiple teams and engineering managers",
                "Own budget and resource allocation",
                "Represent engineering to executive leadership"
            ]
        }

        responsibilities = base_responsibilities.get(role_category, [])[:5]
        if seniority in seniority_additions:
            responsibilities.extend(self._random.sample(seniority_additions[seniority], 2))

        return responsibilities

    def _generate_qualifications(
        self,
        seniority: str,
        years_experience: int,
        required_skills: list[str],
        role_category: str
    ) -> list[str]:
        """
        Generate role qualifications.

        Args:
            seniority: Seniority level
            years_experience: Required years of experience
            required_skills: Required skills
            role_category: Category of the role

        Returns:
            List of qualification strings
        """
        qualifications = [
            f"{years_experience}+ years of software engineering experience",
            f"Strong proficiency in {', '.join(required_skills[:2])}",
            "Experience building production systems at scale"
        ]

        if role_category in ["ml_engineer", "data_scientist"]:
            qualifications.extend([
                "Strong foundation in statistics and machine learning theory",
                "Experience with ML frameworks (TensorFlow, PyTorch)",
                "Track record of deploying models to production"
            ])

        if seniority in ["Senior", "Staff", "Principal"]:
            qualifications.extend([
                "Demonstrated ability to mentor and lead technical projects",
                "Experience with system design and architecture"
            ])

        if seniority in ["Staff", "Principal", "Director"]:
            qualifications.extend([
                "Experience driving technical strategy across teams",
                "Strong communication skills with executive presence"
            ])

        # Education requirements vary
        if role_category == "data_scientist" or self._random.random() < 0.3:
            qualifications.append(
                "MS or PhD in Computer Science, Statistics, or related field preferred"
            )
        else:
            qualifications.append(
                "BS in Computer Science or equivalent practical experience"
            )

        return qualifications

    def _generate_description(
        self,
        title: str,
        company: str,
        seniority: str,
        required_skills: list[str],
        role_category: str
    ) -> str:
        """
        Generate comprehensive job description.

        Args:
            title: Job title
            company: Company name
            seniority: Seniority level
            required_skills: Required skills
            role_category: Role category

        Returns:
            Full job description text
        """
        # Generate team context
        context_template = self._random.choice(TEAM_CONTEXTS)
        context = context_template.format(
            industry=self._random.choice(INDUSTRIES),
            product=self._random.choice(PRODUCTS),
            mission=self._random.choice(MISSIONS)
        )

        # Role-specific descriptions
        role_intros = {
            "ml_engineer": (
                f"We're seeking a {seniority} {title} to join our team and help build "
                f"cutting-edge machine learning systems. You'll work on problems ranging from "
                f"large-scale model training to real-time inference optimization."
            ),
            "data_scientist": (
                f"We're looking for a {seniority} {title} who can turn data into actionable "
                f"insights. You'll work closely with product teams to design experiments, "
                f"build predictive models, and drive data-informed decision making."
            ),
            "backend_engineer": (
                f"We need a {seniority} {title} to help scale our backend infrastructure. "
                f"You'll design and build robust services that power our products and handle "
                f"millions of requests daily."
            ),
            "fullstack_engineer": (
                f"Join us as a {seniority} {title} to build end-to-end product features. "
                f"You'll work across the stack, from pixel-perfect UIs to scalable APIs, "
                f"delivering complete solutions to our users."
            ),
            "engineering_manager": (
                f"We're seeking a {seniority} {title} to lead and grow our engineering team. "
                f"You'll shape technical strategy, build a world-class team, and drive "
                f"delivery of critical initiatives."
            )
        }

        intro = role_intros.get(role_category, f"We're hiring a {seniority} {title}.")

        description = f"""## About {company}

{company} is a {context}. We're building the future of technology and looking for
exceptional people to join our mission.

## About the Role

{intro}

As a member of our team, you'll have the opportunity to work with
{', '.join(required_skills[:3])} and other cutting-edge technologies. We value
ownership, collaboration, and continuous learning.

## Why Join Us?

- Work on challenging problems with real-world impact
- Learn from and collaborate with world-class engineers
- Competitive compensation with equity upside
- Flexible work arrangements and excellent benefits
- Fast-paced environment with opportunities for rapid growth

## Our Tech Stack

We leverage modern technologies including {', '.join(required_skills[:4])}
and more. We're always evaluating new tools and approaches to stay at the
forefront of technology.

## The Team

You'll join a team of {self._random.randint(5, 15)} engineers who are passionate
about building great products. We foster an inclusive environment where
everyone's voice is heard and valued.
"""
        return description

    def generate_role(self, role_id: int) -> Role:
        """
        Generate a single complete role.

        Args:
            role_id: Numeric ID for the role

        Returns:
            Complete Role object
        """
        # Select role category and attributes
        role_category = self._random.choice(list(ROLE_TEMPLATES.keys()))
        template = ROLE_TEMPLATES[role_category]

        title = self._random.choice(template["titles"])
        seniority = self._random.choice(SENIORITY_LEVELS)

        # Adjust title based on seniority
        if seniority == "Junior" and "Junior" not in title and "Associate" not in title:
            title = f"Junior {title}"
        elif seniority == "Senior" and "Senior" not in title:
            title = f"Senior {title}"
        elif seniority == "Staff" and "Staff" not in title:
            title = f"Staff {title}"
        elif seniority == "Principal" and "Principal" not in title:
            title = f"Principal {title}"

        # Required years based on seniority
        yoe_map = {
            "Junior": self._random.randint(0, 2),
            "Mid-Level": self._random.randint(2, 4),
            "Senior": self._random.randint(5, 8),
            "Staff": self._random.randint(8, 12),
            "Principal": self._random.randint(12, 15),
            "Director": self._random.randint(10, 20)
        }
        years_experience = yoe_map.get(seniority, 5)

        # Select skills
        required_skills = self._random.choice(template["required_skills"])
        nice_to_have = self._random.sample(
            template["nice_to_have"],
            min(4, len(template["nice_to_have"]))
        )

        company = self._random.choice(COMPANIES)
        location = self._random.choice(LOCATIONS)

        # Generate benefits
        benefits = [
            "Competitive salary and equity",
            f"{self._random.choice(['Unlimited', 'Generous'])} PTO",
            "Comprehensive health, dental, and vision insurance",
            f"${self._random.randint(1, 5)}K annual learning & development budget",
            "401(k) with company match",
            "Flexible work arrangements",
            f"${self._random.randint(100, 200)}/month wellness stipend",
            "Parental leave",
            "Home office setup allowance"
        ]

        return Role(
            id=f"ROLE_{role_id:03d}",
            title=title,
            company=company,
            location=location,
            seniority=seniority,
            required_skills=required_skills,
            nice_to_have_skills=nice_to_have,
            years_experience_required=years_experience,
            description=self._generate_description(
                title, company, seniority, required_skills, role_category
            ),
            responsibilities=self._generate_responsibilities(
                role_category, seniority, required_skills
            ),
            qualifications=self._generate_qualifications(
                seniority, years_experience, required_skills, role_category
            ),
            benefits=self._random.sample(benefits, self._random.randint(5, 7)),
            salary_range=self._get_salary_range(seniority, role_category)
        )

    def generate_roles(self, count: int = 20) -> list[Role]:
        """
        Generate multiple role postings.

        Args:
            count: Number of roles to generate

        Returns:
            List of Role objects
        """
        logger.info(f"Generating {count} synthetic roles...")
        roles = [self.generate_role(i + 1) for i in range(count)]
        logger.info(f"Successfully generated {len(roles)} roles")
        return roles

    def save_roles(
        self,
        roles: list[Role],
        output_path: Path | str
    ) -> None:
        """
        Save roles to a JSON file.

        Args:
            roles: List of roles to save
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [r.to_dict() for r in roles]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(roles)} roles to {output_path}")
