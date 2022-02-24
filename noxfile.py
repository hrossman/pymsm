import nox

locations = "src", "tests", "noxfile.py"


@nox.session(python=["3.8"])
def tests(session):
    session.run("poetry", "install", external=True)
    session.run("pytest")


@nox.session(python="3.8")
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)
