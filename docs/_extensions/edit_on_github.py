
import os
import warnings


def get_github_url(app, view, path):
    if path.endswith(".ipynb"):
        return app.config.github_nb_repo, "/"
    return 'https://github.com/{project}/{view}/{branch}/{path}'.format(
        project=app.config.edit_on_github_project,
        view=view,
        branch=app.config.edit_on_github_branch,
        path=path)


def html_page_context(app, pagename, templatename, context, doctree):
    if templatename != "page.html":
        return

    if not app.config.github_repo:
        warnings.warn("`github_repo `not specified")
        return

    if not app.config.github_nb_repo:
        nb_repo = f"{app.config.github_repo}_notebooks"
        warnings.warn("`github_nb_repo `not specified. Setting to `{nb_repo}`")
        app.config.github_nb_repo = nb_repo

    path = os.path.relpath(doctree.get("source"), app.builder.srcdir)
    show_url = get_github_url(app, 'blob', path)
    edit_url = get_github_url(app, 'edit', path)

    context['show_on_github_url'] = show_url
    context['edit_on_github_url'] = edit_url

    # For sphinx_rtd_theme.
    context["display_github"] = True
    context["github_user"] = "theislab"
    context["github_version"] = "master"
    context["github_repo"] = app.config.edit_on_github_project.split('/')[1]
    context['source_suffix'] = app.config.source_suffix[0]


def setup(app):
    app.add_config_value("github_nb_repo", "", True)
    app.add_config_value("github_repo", "", True)
    app.connect("html-page-context", html_page_context)
