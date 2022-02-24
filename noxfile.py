import nox
import tempfile
from nox.sessions import Session

locations = "src", "tests", "noxfile.py"


def install_with_constraints(session, *args, **kwargs):
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)

@nox.session(python=["3.8"])
def tests(session):
    session.run("poetry", "install", external=True)
    # session.run("pytest", "--cov")


# @nox.session(python=["3.8"])
# def lint(session):
#     args = session.posargs or locations
#     session.install("flake8")
#     session.run("flake8", *args)


@nox.session(python="3.8")
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)


# @nox.session(python="3.8")
# def coverage(session: Session) -> None:
#     """Upload coverage data."""
#     install_with_constraints(session, "coverage[toml]", "codecov")
#     session.run("coverage", "xml", "--fail-under=0")
#     session.run("codecov", *session.posargs)
