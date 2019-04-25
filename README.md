This is the source code for my personal website, which is currently hosted using github pages. It was created using the [`gatsby-starter-blog`](https://github.com/gatsbyjs/gatsby-starter-blog).

## Setup

To build and develop, you'll need a recent version `npm` and `node`. If you're on macOS or Linux, you likely already have it installed, but if not I suggest using [nvm](https://github.com/nvm-sh/nvm). I have not attempted to use NPM on Windows.

To get started, simply run the following from inside the project folder:

```
npm install --dev
```

This will create a `node_modules` folder with all the project dependencies.

## Development

To start a development server, run

```
npm start
```

This will let you visit your site at `localhost:8000`, and will hot-reload the page as you make changes.

## Building

To create the actual static assets inside of the `./public` folder, run:

```
npm run build
```

This will process all the blog posts, assets and javascript files and turn them into static HTML + JS files (and assets) that you can then host wherever you want.

### Github Pages

I'm using the [free static site hosting](https://pages.github.com/) that github offers to host this. If you'd like to do the same, deployment is really easy. First, add the target repository as a remote called `pages`

```
git remote add pages git@github.com:username/username.github.io.git
```

Then, whenever you want to build and deploy your site, run

```
npm run deploy
```

## Blog Posts

The blog posts are written in Markdown inside of `./blog/post_title/index.md`. Make sure to include a front matter (use the existing blog posts as an example).

## Other important files

If you're going to use this for your own personal website, make sure you change `gatsby-config.js`, `./src/pages/about.js` and `./src/components/sidebar.js` to scrub all my links and name. I didn't really write this with the intention of other people using it so most content in the layout is just hardcoded.