# Contributing to Omnes

Thank you for your interest in contributing to **Omnes**! We welcome all kinds of contributions: new features, bug fixes, documentation, tests, and more.

---

## Contribution Workflow

### 1. Fork the Repository

* Visit: [Omnes GitHub Repo](https://github.com/ntatrishvili/Omnes)
* Click **"Fork"** to create your copy.

### 2. Clone Your Fork Locally

```bash
git clone https://github.com/<your-username>/Omnes.git
cd Omnes
```

### 3. Set Up Your Environment

We recommend using [Poetry](https://python-poetry.org/) for dependency management:

```bash
poetry install
```

### 4. Pick an Open Issue

* Browse issues: [GitHub Issues](https://github.com/ntatrishvili/Omnes/issues)
* Comment to claim one or suggest a new enhancement.

### 5. Create a Branch

```bash
git checkout -b <issue-number>-short-description
```

### 6. Make Your Changes

* Modify the code based on the issue
* **Always add unit tests** for new logic
* Follow project conventions

### 7. Run Tests

```bash
pytest
```

### 8. Format & Secure Code

```bash
black .
bandit -r . --exclude ./tests/
```

### 9. Commit and Push

```bash
git add .
git commit -m "Fix #<issue-number>: <brief description>"
git push origin <branch-name>
```

### 10. Open a Pull Request

* Title: `Fix #<issue-number>: <issue title>`
* Fill out the pull request checklist
* Add reviewers: **Nia Tatrishvili** and **Lilla Barancsuk**

---

## Pull Request Checklist

* [ ] My code follows the style guidelines
* [ ] I have performed a self-review
* [ ] I have added tests
* [ ] I have run `black` and `bandit`
* [ ] I have linked the related GitHub issue
* [ ] I have added Nia Tatrishvili and Lilla Barancsuk as reviewers

---

## Review Process

1. After opening a PR, GitHub will notify the reviewers.
2. **Nia Tatrishvili** and **Lilla Barancsuk** will review your PR and leave comments or approve.
3. You may be asked to revise code for clarity, design, or test coverage.
4. Once approved and all checks pass, the PR will be merged by a maintainer.

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

Thanks again for contributing to **Omnes**!
