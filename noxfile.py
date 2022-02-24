import nox


@nox.session(python=["3.8"])
def tests(session):
    session.run("poetry", "install", external=True)
    session.run("pytest", "--cov")


locations = "src", "tests", "noxfile.py"


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


@nox.session(python="3.8")
def coverage(session: Session) -> None:
    """Upload coverage data."""
    install_with_constraints(session, "coverage[toml]", "codecov")
    session.run("coverage", "xml", "--fail-under=0")
    session.run("codecov", *session.posargs)
