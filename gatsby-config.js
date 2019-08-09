module.exports = {
  siteMetadata: {
    title: `Bruno Opsenica's Blog`,
    author: `Bruno Opsenica`,
    description: `A blog about learning 3D programming`,
    siteUrl: `https://bruop.github.com/`,
    social: {
      twitter: `bruops`,
    },
  },
  plugins: [
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        path: `${__dirname}/content/blog`,
        name: `blog`,
      },
    },
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        path: `${__dirname}/content/assets`,
        name: `assets`,
      },
    },
    {
      resolve: `gatsby-transformer-remark`,
      options: {
        plugins: [
          {
            resolve: "gatsby-remark-images-grid",
            options: {
              gridGap: "10px",
              margin: "20px auto",
            },
          },
          {
            resolve: `gatsby-remark-images`,
            options: {
              quality: 90,
              maxWidth: 720,
            },
          },
          {
            resolve: `gatsby-remark-responsive-iframe`,
            options: {
              wrapperStyle: `margin-bottom: 1.0725rem`,
            },
          },
          {
            resolve: `gatsby-remark-prismjs`,
          },
          `gatsby-remark-copy-linked-files`,
          `gatsby-remark-smartypants`,
          {
            resolve: `gatsby-remark-katex`,
            options: {
              strict: `ignore`,
            },
          },
        ],
      },
    },
    `gatsby-transformer-sharp`,
    `gatsby-plugin-sharp`,
    `gatsby-plugin-feed`,
    `gatsby-plugin-styled-components`,
    {
      resolve: `gatsby-plugin-manifest`,
      options: {
        name: `Bruno Opsenica's Blog`,
        short_name: `BruOp's Blog`,
        start_url: `/`,
        background_color: `#ffffff`,
        theme_color: `#1ca086`,
        display: `minimal-ui`,
        icon: `content/assets/bunny.jpg`,
      },
    },
    `gatsby-plugin-offline`,
    `gatsby-plugin-react-helmet`,
    {
      resolve: `gatsby-plugin-typography`,
      options: {
        pathToConfigModule: `src/utils/typography`,
      },
    },
    {
      resolve: "@debiki/gatsby-plugin-talkyard",
      options: {
        talkyardServerUrl: "https://comments-for-bruop-github-io.talkyard.net",
      },
    },
  ],
};
