"""
Candidate Data Generator Module.

This module generates synthetic candidate data including profiles, resumes,
and interview transcripts for testing the Talent RAG system.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field, asdict

from faker import Faker
from loguru import logger


# Technical skills pool organized by domain
SKILL_POOLS = {
    "programming_languages": [
        "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust",
        "Ruby", "Kotlin", "Swift", "Scala", "R", "MATLAB", "Julia"
    ],
    "web_frameworks": [
        "React", "Angular", "Vue.js", "Next.js", "Django", "Flask", "FastAPI",
        "Express.js", "Spring Boot", "Ruby on Rails", "ASP.NET Core"
    ],
    "data_ml": [
        "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy", "Keras",
        "XGBoost", "LightGBM", "Hugging Face", "OpenCV", "SpaCy", "NLTK",
        "MLflow", "Kubeflow", "Ray", "Dask"
    ],
    "cloud_devops": [
        "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform", "Ansible",
        "Jenkins", "GitHub Actions", "CircleCI", "ArgoCD", "Prometheus", "Grafana"
    ],
    "databases": [
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
        "DynamoDB", "Neo4j", "ClickHouse", "Snowflake", "BigQuery"
    ],
    "ml_specializations": [
        "NLP", "Computer Vision", "Reinforcement Learning", "Deep Learning",
        "LLMs", "RAG Systems", "Recommendation Systems", "Time Series Analysis",
        "Anomaly Detection", "A/B Testing", "Causal Inference"
    ]
}

UNIVERSITIES = [
    "MIT", "Stanford University", "Carnegie Mellon University", "UC Berkeley",
    "Georgia Tech", "University of Michigan", "Cornell University",
    "University of Washington", "UCLA", "UT Austin", "UIUC",
    "University of Toronto", "ETH Zurich", "Oxford University",
    "Cambridge University", "IIT Bombay", "IIT Delhi", "NUS Singapore"
]

DEGREES = [
    ("Bachelor's", "Computer Science"),
    ("Bachelor's", "Software Engineering"),
    ("Bachelor's", "Electrical Engineering"),
    ("Master's", "Computer Science"),
    ("Master's", "Machine Learning"),
    ("Master's", "Data Science"),
    ("Master's", "Artificial Intelligence"),
    ("Ph.D.", "Computer Science"),
    ("Ph.D.", "Machine Learning"),
    ("Ph.D.", "Statistics")
]

COMPANIES = [
    "Google", "Meta", "Amazon", "Microsoft", "Apple", "Netflix", "Uber",
    "Airbnb", "Stripe", "Databricks", "Snowflake", "OpenAI", "Anthropic",
    "DeepMind", "NVIDIA", "Tesla", "Palantir", "Two Sigma", "Citadel",
    "Jane Street", "Goldman Sachs", "JPMorgan", "Bloomberg", "Salesforce",
    "Adobe", "VMware", "Coinbase", "Robinhood", "Figma", "Notion"
]

SENIORITY_LEVELS = ["Junior", "Mid-Level", "Senior", "Staff", "Principal", "Director"]

LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
    "Boston, MA", "Los Angeles, CA", "Denver, CO", "Chicago, IL",
    "Atlanta, GA", "San Diego, CA", "Portland, OR", "Miami, FL",
    "Remote", "London, UK", "Toronto, Canada", "Berlin, Germany"
]


@dataclass
class Education:
    """Education record for a candidate."""
    degree: str
    field: str
    university: str
    graduation_year: int


@dataclass
class Experience:
    """Work experience record for a candidate."""
    company: str
    title: str
    duration_years: float
    description: str


@dataclass
class Candidate:
    """
    Complete candidate profile.

    Attributes:
        id: Unique candidate identifier
        name: Full name
        email: Contact email
        location: Current location
        years_of_experience: Total years of professional experience
        skills: List of technical skills
        seniority: Seniority level
        education: List of education records
        experience: List of work experience records
        resume_text: Full resume text
        interview_transcript: Technical and behavioral interview Q&A
        created_at: Timestamp of record creation
    """
    id: str
    name: str
    email: str
    location: str
    years_of_experience: int
    skills: list[str]
    seniority: str
    education: list[dict[str, Any]]
    experience: list[dict[str, Any]]
    resume_text: str
    interview_transcript: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert candidate to dictionary representation."""
        return asdict(self)


class CandidateGenerator:
    """
    Generator for synthetic candidate data.

    This class generates realistic candidate profiles with resumes and
    interview transcripts for testing recruitment AI systems.

    Attributes:
        faker: Faker instance for generating random data
        seed: Random seed for reproducibility
    """

    def __init__(self, seed: int | None = None):
        """
        Initialize the candidate generator.

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
        logger.info(f"CandidateGenerator initialized with seed={seed}")

    def _generate_skills(self, seniority: str, specialization: str) -> list[str]:
        """
        Generate a realistic skill set based on seniority and specialization.

        Args:
            seniority: Candidate's seniority level
            specialization: Primary domain specialization

        Returns:
            List of skills appropriate for the candidate profile
        """
        # Base skill count varies by seniority
        seniority_skill_counts = {
            "Junior": (4, 7),
            "Mid-Level": (6, 10),
            "Senior": (8, 14),
            "Staff": (10, 16),
            "Principal": (12, 18),
            "Director": (10, 15)
        }
        min_skills, max_skills = seniority_skill_counts.get(seniority, (6, 10))
        num_skills = self._random.randint(min_skills, max_skills)

        skills = []

        # Add programming languages (2-4)
        skills.extend(self._random.sample(
            SKILL_POOLS["programming_languages"],
            min(self._random.randint(2, 4), len(SKILL_POOLS["programming_languages"]))
        ))

        # Add specialization-specific skills
        if specialization in ["ml_engineer", "data_scientist"]:
            skills.extend(self._random.sample(
                SKILL_POOLS["data_ml"],
                min(self._random.randint(3, 6), len(SKILL_POOLS["data_ml"]))
            ))
            skills.extend(self._random.sample(
                SKILL_POOLS["ml_specializations"],
                min(self._random.randint(2, 4), len(SKILL_POOLS["ml_specializations"]))
            ))
        elif specialization in ["backend_engineer", "fullstack_engineer"]:
            skills.extend(self._random.sample(
                SKILL_POOLS["web_frameworks"],
                min(self._random.randint(2, 4), len(SKILL_POOLS["web_frameworks"]))
            ))
            skills.extend(self._random.sample(
                SKILL_POOLS["databases"],
                min(self._random.randint(2, 4), len(SKILL_POOLS["databases"]))
            ))
        else:
            # General mix
            skills.extend(self._random.sample(
                SKILL_POOLS["web_frameworks"],
                min(self._random.randint(1, 3), len(SKILL_POOLS["web_frameworks"]))
            ))

        # Add cloud/devops skills
        skills.extend(self._random.sample(
            SKILL_POOLS["cloud_devops"],
            min(self._random.randint(2, 4), len(SKILL_POOLS["cloud_devops"]))
        ))

        # Deduplicate and limit
        skills = list(set(skills))[:num_skills]
        return skills

    def _generate_education(self, seniority: str) -> list[Education]:
        """
        Generate education history based on seniority.

        Args:
            seniority: Candidate's seniority level

        Returns:
            List of Education records
        """
        education = []
        current_year = datetime.now().year

        # Higher seniority tends to have more advanced degrees
        if seniority in ["Principal", "Director", "Staff"]:
            # More likely to have advanced degrees
            if self._random.random() < 0.4:
                degree, field_of_study = self._random.choice([d for d in DEGREES if "Ph.D." in d[0]])
                education.append(Education(
                    degree=degree,
                    field=field_of_study,
                    university=self._random.choice(UNIVERSITIES),
                    graduation_year=current_year - self._random.randint(8, 20)
                ))
            if self._random.random() < 0.7:
                degree, field_of_study = self._random.choice([d for d in DEGREES if "Master's" in d[0]])
                education.append(Education(
                    degree=degree,
                    field=field_of_study,
                    university=self._random.choice(UNIVERSITIES),
                    graduation_year=current_year - self._random.randint(10, 22)
                ))

        # Everyone has at least a bachelor's
        degree, field_of_study = self._random.choice([d for d in DEGREES if "Bachelor's" in d[0]])
        base_grad_year = {
            "Junior": self._random.randint(0, 3),
            "Mid-Level": self._random.randint(3, 6),
            "Senior": self._random.randint(6, 12),
            "Staff": self._random.randint(10, 18),
            "Principal": self._random.randint(12, 22),
            "Director": self._random.randint(15, 25)
        }
        education.append(Education(
            degree=degree,
            field=field_of_study,
            university=self._random.choice(UNIVERSITIES),
            graduation_year=current_year - base_grad_year.get(seniority, 5)
        ))

        return education

    def _generate_experience(
        self,
        seniority: str,
        years_of_experience: int,
        skills: list[str]
    ) -> list[Experience]:
        """
        Generate work experience history.

        Args:
            seniority: Candidate's seniority level
            years_of_experience: Total years of experience
            skills: Candidate's skills for context

        Returns:
            List of Experience records
        """
        experiences = []
        remaining_years = years_of_experience

        # Determine number of positions based on experience
        num_positions = min(self._random.randint(2, 5), max(1, years_of_experience // 2))

        titles_by_seniority = {
            "Junior": ["Software Engineer", "Junior Developer", "Associate Engineer"],
            "Mid-Level": ["Software Engineer II", "Developer", "ML Engineer"],
            "Senior": ["Senior Software Engineer", "Senior ML Engineer", "Tech Lead"],
            "Staff": ["Staff Engineer", "Staff ML Engineer", "Engineering Manager"],
            "Principal": ["Principal Engineer", "Distinguished Engineer", "ML Architect"],
            "Director": ["Director of Engineering", "VP Engineering", "CTO"]
        }

        for i in range(num_positions):
            # Allocate years to this position
            if i == num_positions - 1:
                duration = remaining_years
            else:
                duration = self._random.uniform(1, min(4, remaining_years - (num_positions - i - 1)))

            remaining_years -= duration

            # Generate title based on position in career
            if i == 0:  # Current position
                title = self._random.choice(titles_by_seniority.get(seniority, ["Engineer"]))
            else:
                # Earlier positions have lower seniority
                earlier_seniority = SENIORITY_LEVELS[
                    max(0, SENIORITY_LEVELS.index(seniority) - i - 1)
                ]
                title = self._random.choice(titles_by_seniority.get(earlier_seniority, ["Engineer"]))

            # Generate description using skills
            skill_mentions = self._random.sample(skills, min(3, len(skills)))
            descriptions = [
                f"Led development of {self.faker.bs()} using {', '.join(skill_mentions[:2])}.",
                f"Built scalable systems handling {self._random.randint(100, 10000)}K+ requests/day.",
                "Collaborated with cross-functional teams to deliver key product features.",
                f"Mentored {self._random.randint(2, 8)} junior engineers on best practices.",
                f"Designed and implemented {self.faker.catch_phrase().lower()} infrastructure.",
                f"Reduced system latency by {self._random.randint(20, 60)}% through optimization.",
                "Contributed to open-source projects and internal tooling.",
            ]

            experiences.append(Experience(
                company=self._random.choice(COMPANIES),
                title=title,
                duration_years=round(duration, 1),
                description=" ".join(self._random.sample(descriptions, self._random.randint(2, 4)))
            ))

            if remaining_years <= 0:
                break

        return experiences

    def _generate_resume_text(
        self,
        name: str,
        seniority: str,
        skills: list[str],
        education: list[Education],
        experience: list[Experience],
        specialization: str
    ) -> str:
        """
        Generate comprehensive resume text (300+ words).

        Args:
            name: Candidate name
            seniority: Seniority level
            skills: List of skills
            education: Education history
            experience: Work experience
            specialization: Domain specialization

        Returns:
            Full resume text as a string
        """
        # Professional summary
        specialization_summaries = {
            "ml_engineer": "Experienced Machine Learning Engineer with expertise in "
                          "building production ML systems, developing deep learning models, "
                          "and deploying scalable AI solutions.",
            "data_scientist": "Data Scientist with strong background in statistical analysis, "
                             "predictive modeling, and deriving actionable insights from "
                             "complex datasets.",
            "backend_engineer": "Backend Engineer specialized in building high-performance, "
                               "scalable distributed systems and microservices architectures.",
            "fullstack_engineer": "Full-stack Engineer with comprehensive experience across "
                                 "frontend and backend technologies, delivering end-to-end "
                                 "product solutions."
        }

        summary = specialization_summaries.get(
            specialization,
            "Software Engineer with proven track record of delivering impactful solutions."
        )

        resume_parts = [
            f"# {name}",
            f"## {seniority} {specialization.replace('_', ' ').title()}",
            "",
            "## Professional Summary",
            f"{summary} Passionate about {self.faker.bs()} and driving technical excellence. "
            f"Strong communicator with experience leading cross-functional initiatives and "
            f"mentoring engineering teams. Committed to continuous learning and staying "
            f"current with industry best practices and emerging technologies.",
            "",
            "## Technical Skills",
            f"**Core Technologies:** {', '.join(skills[:len(skills)//2])}",
            f"**Additional Skills:** {', '.join(skills[len(skills)//2:])}",
            "",
            "## Professional Experience",
        ]

        for exp in experience:
            resume_parts.extend([
                f"### {exp.title} at {exp.company}",
                f"*{exp.duration_years} years*",
                "",
                exp.description,
                "",
                f"Key achievements include improving team velocity by {self._random.randint(15, 40)}%, "
                f"implementing automated testing pipelines that caught {self._random.randint(50, 200)} "
                f"bugs before production, and contributing to architectural decisions that "
                f"improved system reliability to {self._random.uniform(99.5, 99.99):.2f}% uptime.",
                ""
            ])

        resume_parts.append("## Education")
        for edu in education:
            resume_parts.extend([
                f"### {edu.degree} in {edu.field}",
                f"*{edu.university}, {edu.graduation_year}*",
                ""
            ])

        resume_parts.extend([
            "## Notable Projects",
            f"- Developed {self.faker.catch_phrase().lower()} system that processed "
            f"{self._random.randint(1, 100)}M+ records daily with sub-second latency.",
            f"- Created open-source library for {self.faker.bs()} with "
            f"{self._random.randint(100, 5000)}+ GitHub stars.",
            f"- Implemented ML pipeline reducing prediction time by {self._random.randint(30, 70)}% "
            f"while maintaining {self._random.randint(92, 99)}% accuracy.",
            "- Led migration of legacy monolith to microservices, reducing deployment time "
            "from hours to minutes.",
            "",
            "## Publications & Talks",
            f"- Published research on {self._random.choice(SKILL_POOLS['ml_specializations'])} "
            f"at {self._random.choice(['NeurIPS', 'ICML', 'ACL', 'CVPR', 'KDD'])} "
            f"{datetime.now().year - self._random.randint(1, 4)}.",
            f"- Speaker at {self._random.choice(['PyCon', 'KubeCon', 'QCon', 'StrangeLoop'])} "
            f"on {self.faker.bs()}.",
        ])

        return "\n".join(resume_parts)

    def _generate_interview_transcript(
        self,
        name: str,
        seniority: str,
        skills: list[str],
        specialization: str
    ) -> str:
        """
        Generate interview transcript with technical and behavioral Q&A.

        Args:
            name: Candidate name
            seniority: Seniority level
            skills: Candidate skills
            specialization: Domain specialization

        Returns:
            Interview transcript text
        """
        transcript_parts = [
            f"# Interview Transcript - {name}",
            f"**Date:** {self.faker.date_between(start_date='-90d', end_date='today')}",
            f"**Position:** {seniority} {specialization.replace('_', ' ').title()}",
            "",
            "---",
            "",
            "## Technical Interview",
            ""
        ]

        # Technical questions based on skills
        tech_questions = [
            (
                f"Can you explain your experience with {self._random.choice(skills)}?",
                f"I have {self._random.randint(2, 8)} years of hands-on experience with "
                f"{self._random.choice(skills)}. In my previous role at {self._random.choice(COMPANIES)}, "
                f"I used it extensively for building {self.faker.bs()}. I particularly focused on "
                f"optimizing performance and ensuring code quality through comprehensive testing. "
                f"One notable project involved reducing latency by {self._random.randint(30, 60)}% "
                f"through careful profiling and optimization."
            ),
            (
                "Describe a challenging technical problem you solved recently.",
                f"Recently, I tackled a complex scaling issue where our system was experiencing "
                f"degraded performance under high load. After thorough investigation, I identified "
                f"that the bottleneck was in our database queries. I implemented a combination of "
                f"query optimization, caching with {self._random.choice(['Redis', 'Memcached'])}, and "
                f"horizontal scaling that improved throughput by {self._random.randint(3, 10)}x while "
                f"reducing p99 latency from {self._random.randint(500, 2000)}ms to under 100ms."
            ),
            (
                "How do you approach system design for large-scale applications?",
                f"I follow a structured approach starting with understanding requirements and "
                f"constraints. I consider factors like expected load, data consistency requirements, "
                f"and failure modes. For a recent project serving {self._random.randint(10, 100)}M users, "
                f"I designed a microservices architecture using {self._random.choice(skills)} with "
                f"event-driven communication via Kafka. I ensured high availability through "
                f"multi-region deployment and implemented circuit breakers for resilience."
            ),
            (
                f"What's your experience with {self._random.choice(SKILL_POOLS['cloud_devops'])}?",
                f"I've been working with cloud technologies for {self._random.randint(3, 10)} years. "
                f"I'm particularly experienced with infrastructure as code using Terraform, "
                f"container orchestration with Kubernetes, and implementing CI/CD pipelines. "
                f"I've led cloud migrations that reduced infrastructure costs by "
                f"{self._random.randint(20, 50)}% while improving reliability and deployment velocity."
            ),
        ]

        # Add ML-specific questions for ML roles
        if specialization in ["ml_engineer", "data_scientist"]:
            tech_questions.extend([
                (
                    "Explain your approach to MLOps and model deployment.",
                    f"I believe in treating ML systems as software products with proper versioning, "
                    f"testing, and monitoring. I've implemented MLOps pipelines using "
                    f"{self._random.choice(['MLflow', 'Kubeflow', 'SageMaker'])} for experiment tracking, "
                    f"model registry, and automated deployment. I ensure models are monitored for "
                    f"drift and have automated retraining pipelines. In my last project, this "
                    f"reduced time-to-production from weeks to hours."
                ),
                (
                    "How do you handle model performance issues in production?",
                    f"I implement comprehensive monitoring including prediction latency, throughput, "
                    f"and business metrics. For model quality, I track feature drift and prediction "
                    f"distribution shifts. When I detected a {self._random.randint(5, 15)}% accuracy drop "
                    f"in a production model, I quickly identified a data quality issue in an upstream "
                    f"system and implemented validation checks that prevented similar issues."
                ),
            ])

        # Select 4-5 questions
        selected_tech = self._random.sample(tech_questions, min(5, len(tech_questions)))

        for q, a in selected_tech:
            transcript_parts.extend([
                f"**Q: {q}**",
                "",
                f"*{name}:* {a}",
                ""
            ])

        transcript_parts.extend([
            "---",
            "",
            "## Behavioral Interview",
            ""
        ])

        behavioral_questions = [
            (
                "Tell me about a time when you had to influence without authority.",
                f"At {self._random.choice(COMPANIES)}, I identified a critical technical debt issue that "
                f"was slowing down the team. I gathered data on the impact, created a compelling "
                f"presentation, and proposed a phased remediation plan. By showing the ROI in terms "
                f"of developer productivity and reduced incident count, I convinced leadership to "
                f"allocate {self._random.randint(2, 4)} sprints for the initiative. The result was a "
                f"{self._random.randint(25, 50)}% reduction in time spent on maintenance."
            ),
            (
                "Describe a situation where you had a conflict with a teammate.",
                "I had a disagreement with a colleague about the architecture for a new feature. "
                "Instead of escalating, I suggested we both prototype our approaches and evaluate "
                "them against objective criteria. This data-driven approach helped us identify "
                "that a hybrid solution incorporating elements from both designs was optimal. "
                "We maintained a great working relationship and delivered a better solution."
            ),
            (
                "How do you mentor junior team members?",
                f"I believe in a hands-on mentoring approach. I pair program regularly, conduct "
                f"thorough code reviews focused on teaching, and create documentation for common "
                f"patterns. I've mentored {self._random.randint(3, 10)} engineers over my career, several "
                f"of whom have been promoted to senior roles. I also organize internal tech talks "
                f"and encourage my mentees to present, which builds their confidence and visibility."
            ),
            (
                "Tell me about a project that failed and what you learned.",
                "We attempted to build a real-time recommendation system with aggressive timelines. "
                "Despite my concerns about scope, we proceeded and ultimately had to descope "
                "significantly. I learned the importance of pushback early in the process and "
                "now I always advocate for MVP approaches with clear success criteria before "
                "committing to full implementation. This has helped me deliver more consistently."
            ),
        ]

        selected_behavioral = self._random.sample(behavioral_questions, min(3, len(behavioral_questions)))

        for q, a in selected_behavioral:
            transcript_parts.extend([
                f"**Q: {q}**",
                "",
                f"*{name}:* {a}",
                ""
            ])

        transcript_parts.extend([
            "---",
            "",
            "## Interviewer Notes",
            "",
            f"**Technical Assessment:** {self._random.choice(['Strong', 'Very Strong', 'Exceptional'])} - "
            f"Demonstrated deep knowledge of {', '.join(self._random.sample(skills, min(3, len(skills))))}.",
            "",
            f"**Communication:** {self._random.choice(['Clear', 'Articulate', 'Excellent'])} - "
            f"Explained complex concepts effectively.",
            "",
            f"**Culture Fit:** {self._random.choice(['Strong', 'Very Strong', 'Excellent'])} - "
            f"Collaborative mindset, growth-oriented.",
            "",
            f"**Recommendation:** {self._random.choice(['Strong Hire', 'Hire', 'Inclined to Hire'])}",
        ])

        return "\n".join(transcript_parts)

    def generate_candidate(self, candidate_id: int) -> Candidate:
        """
        Generate a single complete candidate profile.

        Args:
            candidate_id: Numeric ID for the candidate

        Returns:
            Complete Candidate object
        """
        # Determine candidate attributes
        seniority = self._random.choice(SENIORITY_LEVELS)
        specialization = self._random.choice([
            "ml_engineer", "data_scientist", "backend_engineer", "fullstack_engineer"
        ])

        # Years of experience correlates with seniority
        yoe_ranges = {
            "Junior": (0, 2),
            "Mid-Level": (2, 5),
            "Senior": (5, 10),
            "Staff": (8, 15),
            "Principal": (12, 20),
            "Director": (15, 25)
        }
        min_yoe, max_yoe = yoe_ranges.get(seniority, (3, 8))
        years_of_experience = self._random.randint(min_yoe, max_yoe)

        name = self.faker.name()
        skills = self._generate_skills(seniority, specialization)
        education = self._generate_education(seniority)
        experience = self._generate_experience(seniority, years_of_experience, skills)

        return Candidate(
            id=f"CAND_{candidate_id:03d}",
            name=name,
            email=self.faker.email(),
            location=self._random.choice(LOCATIONS),
            years_of_experience=years_of_experience,
            skills=skills,
            seniority=seniority,
            education=[asdict(e) for e in education],
            experience=[asdict(e) for e in experience],
            resume_text=self._generate_resume_text(
                name, seniority, skills, education, experience, specialization
            ),
            interview_transcript=self._generate_interview_transcript(
                name, seniority, skills, specialization
            )
        )

    def generate_candidates(self, count: int = 100) -> list[Candidate]:
        """
        Generate multiple candidate profiles.

        Args:
            count: Number of candidates to generate

        Returns:
            List of Candidate objects
        """
        logger.info(f"Generating {count} synthetic candidates...")
        candidates = [self.generate_candidate(i + 1) for i in range(count)]
        logger.info(f"Successfully generated {len(candidates)} candidates")
        return candidates

    def save_candidates(
        self,
        candidates: list[Candidate],
        output_path: Path | str
    ) -> None:
        """
        Save candidates to a JSON file.

        Args:
            candidates: List of candidates to save
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [c.to_dict() for c in candidates]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(candidates)} candidates to {output_path}")
