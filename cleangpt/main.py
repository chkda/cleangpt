import typer

app = typer.Typer(
    name="cleangpt",
    help="An educational llm framework "
)

# TODO: To be removed
@app.command()
def hello(name: str):
    print("Hello " + name)

# TODO: To be removed
@app.command()
def goodbye(name: str):
    print("Goodbye " + name)


if __name__ == "__main__":
    app()